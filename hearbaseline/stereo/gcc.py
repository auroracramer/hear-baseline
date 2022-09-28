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
# Default hop_size in milliseconds
TIMESTAMP_HOP_SIZE = 50
SCENE_HOP_SIZE = 250
STFT_WIN_SIZE = 960
STFT_HOP_SIZE = 480
n_fft = STFT_WIN_SIZE

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

def load_model() -> torch.nn.Module:
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
    hop_size: float = TIMESTAMP_HOP_SIZE,
    win_length: int = STFT_WIN_SIZE,
    hop_length: int = STFT_HOP_SIZE
) -> Tuple[Tensor, Tensor]:
    """
    This function returns embeddings at regular intervals centered at timestamps. Both
    the embeddings and corresponding timestamps (in milliseconds) are returned.

    Args:
        audio: n_sounds x n_channels x n_samples of audio in the range [-1, 1].
        model: Loaded model.
        hop_size: Hop size in milliseconds.
            NOTE: Not required by the HEAR API. We add this optional parameter
            to improve the efficiency of scene embedding.

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

    # Split the input audio signals into frames and then flatten to create a tensor
    # of audio frames that can be batch processed.
    frames, timestamps = frame_audio(
        audio,
        frame_size=16000, # to match size of model audio input (2, )
        hop_size=hop_size,
        sample_rate=STFT2GCCPhat.sample_rate,
    ) # frames: (n_sounds, 2 num_channels, num_frames, 16000 frame_size)
    # Remove channel dimension for mono model
    frames = frames.squeeze(dim=1) # ignore for binaural
    audio_batches, num_channels, num_frames, frame_size = frames.shape
    # frames = frames.flatten(end_dim=1)
    # stack first two dimensions so for mono it's (sounds*frames, frame size)
    frames = frames.reshape(-1, frames.shape[1], frames.shape[3]) # reshape to stack binaural frames

    # convert frames to stft
    stft = Tensor(compute_stft(frames,
                               win_length=win_length,
                               hop_length=hop_length,
                               n_fft=n_fft,
                               pad_mode="constant",
                               center=True))


    # We're using a DataLoader to help with batching of frames
    dataset = torch.utils.data.TensorDataset(stft)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=BATCH_SIZE, shuffle=False, drop_last=False
    )

    # Put the model into eval mode, and not computing gradients while in inference.
    # Iterate over all batches and accumulate the embeddings for each frame.
    model.eval()
    with torch.no_grad():
        embeddings_list = [model(batch[0]) for batch in loader]

    # Concatenate mini-batches back together and unflatten the frames
    # to reconstruct the audio batches
    embeddings = torch.cat(embeddings_list, dim=0)
    embeddings = embeddings.unflatten(0, (audio_batches, num_frames))

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
    embeddings, _ = get_timestamp_embeddings(audio, model, hop_size=SCENE_HOP_SIZE)
    # averaging over frames
    # sounds * frames * features
    embeddings = torch.mean(embeddings, dim=1)

    # Reshape so embeddings for each channel are stacked
    # _, embedding_size = embeddings.shape
    # embeddings = embeddings.reshape(num_sounds, model.num_channels * embedding_size)

    return embeddings