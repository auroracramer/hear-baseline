"""
GCC-phat feature extractor for HEAR.

Adapted from https://github.com/magdalenafuentes/urbansas/blob/main/models/layers.py

"""

from typing import Tuple

import torch
import torch.nn as nn
import torch.fft
from torch import Tensor
from hearbaseline.util import *

# Parameters adapted from Urbansas
STFT_WIN_LENGTH = 960
STFT_HOP_LENGTH = 480
n_fft = STFT_WIN_LENGTH

# Number of frames to batch process for timestamp embeddings
BATCH_SIZE = 512


class STFT2GCCPhat(nn.Module):
    """
    Extract GCC-Phat from multi-channel STFT
    Args:
        max_coeff: maximum number of coefficients, first max_coeff//2 and last max_coeff//2
        kwargs: passed to tf.keras.layers.Layer constructor
    """
    # sample rate and embedding sizes are required model attributes for the HEAR API
    sample_rate = 24000
    num_channels = 2
    embedding_size = 4096
    scene_embedding_size = embedding_size
    timestamp_embedding_size = embedding_size

    def __init__(self, max_coeff: int = None):
        super().__init__()
        self.max_coeff = max_coeff

    def forward(self, inputs: Tensor or Iterable[Tensor, Tensor], **kwargs):
        """
        Args:
            inputs: STFT [, mask]
                STFT: N,ch,T,F complex64 tensor
                    N: batch size
                    ch: number of channels
                    T: time frames
                    F = nfft/2+1
                mask: N,T,F float32 tensor
                    N: number of signals in the batch
                    T: time frames
                    S: max_coeff or nfft
        Returns: N,comb,T,S float32 tensor
                N: number of signals in the batch
                comb: number of channels combinations
                T: time frames
                S: max_coeff or nfft
        """
        num_channels = inputs.shape[1]

        if num_channels < 2:
            raise ValueError(f'GCC-Phat requires at least two input channels')

        out_list = []
        for ch1 in range(num_channels - 1):
            for ch2 in range(ch1 + 1, num_channels):
                x1 = inputs[:, ch1]
                x2 = inputs[:, ch2]
                xcc = torch.angle(x1 * torch.conj(x2))
                xcc = torch.exp(1j * xcc.type(torch.complex64))
                gcc_phat = torch.fft.irfft(xcc)
                if self.max_coeff is not None:
                    gcc_phat = torch.cat([gcc_phat[:, :, -self.max_coeff // 2:], gcc_phat[:, :, :self.max_coeff // 2]],
                                         dim=2)
                out_list.append(gcc_phat)

        return torch.stack(out_list, dim=1)

def load_model(model_file_path: str = "") -> torch.nn.Module:
    """
    Returns a torch.nn.Module that produces embeddings for audio.

    Args:
        model_file_path: Ignored.
    Returns:
        Model
    """

    audio_model = STFT2GCCPhat()

    return audio_model


def get_timestamp_embeddings(
    audio: Tensor,
    model: torch.nn.Module,
) -> Tuple[Tensor, Tensor]:
    """
    This function returns embeddings at regular intervals centered at timestamps. Both
    the embeddings and corresponding timestamps (in milliseconds) are returned.

    Args:
        audio: n_sounds x n_channels x n_samples of audio in the range [-1, 1].
        model: Loaded model.

    Returns:
        - Tensor: embeddings, A float32 Tensor with shape (n_sounds, n_timestamps,
            model.timestamp_embedding_size).
        - Tensor: timestamps, Centered timestamps in milliseconds corresponding
            to each embedding in the output. Shape: (n_sounds, n_timestamps).
    """

    # Assert audio is of correct shape
    if audio.ndim != 3:
        raise ValueError(
            "audio input tensor must be 3D with shape (n_sounds, num_channels, num_samples)"
        )

    if audio.shape[1] != 2:
        raise ValueError(
            "audio input tensor must be binaural"
        )

    # Make sure the correct model type was passed in
    if not isinstance(model, STFT2GCCPhat):
        raise ValueError(
            f"Model must be an instance of {STFT2GCCPhat.__name__}"
        )

    # Send the model to the same device that the audio tensor is on.
    model = model.to(audio.device)

    # n_sounds x n_channels x n_samples
    # compute stft
    stft = Tensor(compute_stft(audio,
                               win_length=STFT_WIN_LENGTH,
                               hop_length=STFT_HOP_LENGTH,
                               n_fft=n_fft,
                               pad_mode="constant",
                               center=True))

    # Put the model into eval mode, and not computing gradients while in inference.
    # Iterate over all batches and accumulate the embeddings for each frame.
    model.eval()
    hop_size = STFT_HOP_LENGTH / float(model.sample_rate)
    with torch.no_grad():
        # embeddings: (N,comb,T,S)
        embeddings = model(stft)

    n_sounds, n_comb, n_time, n_freq = embeddings.shape
    # Swap channel-combination and time dimension
    embeddings.transpose(1, 2)
    # Combine "comb" and "n_freq" dimensions
    embeddings = embeddings.reshape(n_sounds, n_time, n_comb * n_freq)

    # Create time stamps from hop_size
    timestamps = (torch.arange(n_time) * hop_size).repeat(n_sounds, 1)

    return embeddings, timestamps


def get_scene_embeddings(audio: Tensor, model: torch.nn.Module, *args, **kwargs) -> Tensor:
    """
    This function returns a single embedding for each audio clip.

    Args:
        audio: n_sounds x n_channels x n_samples of audio in the range [-1, 1]. All sounds in
            a batch will be padded/trimmed to the same length.
        model: Loaded model.

    Returns:
        - embeddings, A float32 Tensor with shape
            (n_sounds, model.scene_embedding_size).
    """
    num_sounds, _num_channels, num_samples = audio.shape
    if _num_channels != model.num_channels:
        raise ValueError(
            f"audio input tensor must be have {model._n_input_audio} channels, "
            f"but got {_num_channels}"
        )

    # stack bins and channels
    # audio = audio.reshape(num_sounds * _num_channels, 1, num_samples)

    # multi-channel input to multi-channel model
    embeddings, _ = get_timestamp_embeddings(audio, model)
    # averaging over frames
    # sounds * frames * features
    embeddings = torch.mean(embeddings, dim=1)

    # Reshape so embeddings for each channel are stacked
    # _, embedding_size = embeddings.shape
    # embeddings = embeddings.reshape(num_sounds, model.num_channels * embedding_size)

    return embeddings