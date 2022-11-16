"""
Baseline model for HEAR 2021 NeurIPS competition.

This is simply a mel spectrogram followed by random projection.
"""

from collections import OrderedDict
import math
from typing import Optional, Tuple

import librosa
import torch
import torch.fft
from torch import Tensor

from hearbaseline.util import frame_audio, compute_stft

WINDOW_SIZE = 1000 # 1 s window
# Default hop_size in milliseconds
TIMESTAMP_HOP_SIZE = 100
SCENE_HOP_SIZE = 250

# Number of frames to batch process for timestamp embeddings
BATCH_SIZE = 512

import operator as op
from functools import reduce
def nCr(n, r):
    r = min(r, n-r)
    numer = reduce(op.mul, range(n, n-r, -1), 1)
    denom = reduce(op.mul, range(1, r+1), 1)
    return numer // denom  # or / in Python 2


class RandomProjectionMelspecGCCEmbedding(torch.nn.Module):
    """
    Baseline audio embedding model. This model creates mel frequency spectrums with
    256 mel-bands, and then performs a projection to an embedding size of 4096.
    """

    # sample rate and embedding sizes are required model attributes for the HEAR API
    sample_rate = 16000
    num_channels = 2
    embedding_size = 128
    scene_embedding_size = embedding_size
    timestamp_embedding_size = embedding_size
    hop_size = TIMESTAMP_HOP_SIZE

    # These attributes are specific to this baseline model
    #n_fft = 4096
    n_mels = 64
    seed = 0

    n_fft: int = 1024 # 64 ms 
    hop_length: int = 320 # 20 ms
    win_length: int = 640 # 40 ms
    downsample: Optional[int] = None # 4

    # original: [spectra] 4096 -(mel)-> 256 -(projection)-> 4096 
    # new:      [spectrogram] 128 * T * 
    def __init__(self, **model_options):
        super().__init__()
        self.update_model_options(**model_options)
        torch.random.manual_seed(self.seed)

        # Create a Hann window buffer to apply to frames prior to FFT.
        self.register_buffer("window", torch.hann_window(self.n_fft))

        # Create a mel filter buffer.
        mel_scale: Tensor = torch.tensor(
            librosa.filters.mel(self.sample_rate, n_fft=self.n_fft, n_mels=self.n_mels)
        )
        self.register_buffer("mel_scale", mel_scale)

        self._num_frames = int((int((WINDOW_SIZE / 1000.0) * self.sample_rate)) / self.hop_length + 1)
        self._num_combs = nCr(self.num_channels, 2)
        self.input_size = self._num_frames * self.n_mels * (self.num_channels + self._num_combs)

        # Projection matrices.
        normalization = math.sqrt(self.n_mels)
        self.projection = torch.nn.Parameter(
            torch.rand(self.input_size, self.embedding_size) / normalization
        )

    def update_model_options(self, **model_options):
        for k, v in model_options.items():
            if   k == "hop_size":
                self.hop_size = v
            elif k == "seed":
                self.seed = v
            elif k == "embedding_size":
                self.embedding_size = v
                self.scene_embedding_size = v
                self.timestamp_embedding_size = v
            elif k == "sample_rate":
                self.sample_rate = v
            elif k == "n_fft":
                self.n_fft = v
            elif k == "n_mels":
                self.n_mels = v
            elif k == "seed":
                self.seed = v

    def compute_melspec(self, stft: Tensor):
        spec = torch.abs(stft) ** 2.0
        # Apply the mel-scale filter to the magnitude spectrogram
        spec = torch.matmul(spec, self.mel_scale.transpose(0, 1))

        # Downsample
        if self.downsample is not None:
            spec = torch.nn.functional.avg_pool2d(
                spec,
                kernel_size=(self.downsample, self.downsample),
            )
        
        # Apply log1p to spectrogram
        spec = torch.log1p(spec)

        return spec
    
    def compute_gcc(self, stft: Tensor):
        num_channels = stft.shape[1]
        assert num_channels == self.num_channels
        # compute gcc_phat : (N,comb,T,F)
        out_list = []
        for ch1 in range(num_channels - 1):
            for ch2 in range(ch1 + 1, num_channels):
                x1 = stft[:, ch1]
                x2 = stft[:, ch2]
                xcc = torch.angle(x1 * torch.conj(x2))
                xcc = torch.exp(1j * xcc.type(torch.complex64))
                gcc_phat = torch.fft.irfft(xcc)
                out_list.append(gcc_phat)
        gcc_phat = torch.stack(out_list, dim=1)
        # Apply the mel-scale filter to the GCC-PHAT values
        gcc_phat = torch.matmul(gcc_phat, self.mel_scale.transpose(0, 1))

        # Downsample
        if self.downsample is not None:
            gcc_phat = torch.nn.functional.avg_pool2d(
                gcc_phat,
                kernel_size=(self.downsample, self.downsample),
            )
        return gcc_phat

    def preprocess_audio(self, audio: Tensor):
        num_channels = audio.shape[1]
        assert num_channels == self.num_channels

        # compute stft : (N,ch,T,F)
        stft = compute_stft(
            audio,
            self.win_length,
            self.hop_length,
            self.n_fft,
            pad_mode="constant",
            center=True,
        )
        spec = self.compute_melspec(stft)
        gcc_phat = self.compute_gcc(stft)

        # return feat : (N, (ch + nCr(ch, 2)) * T * F)
        return torch.cat([spec, gcc_phat], dim=1).flatten(start_dim=1)


    def forward(self, audio: Tensor):
        x = self.preprocess_audio(audio)

        # Apply projection to get a N dimension embedding
        embedding = x.matmul(self.projection)

        return embedding


def load_model(
    model_file_path: str = "",
    **model_options,
) -> torch.nn.Module:
    """
    Returns a torch.nn.Module that produces embeddings for audio.

    Args:
        model_file_path: Load model checkpoint from this file path. For this baseline,
            if no path is provided then the default random init weights for the
            linear projection layer will be used.
        hop_size: (Optional) Hop size in milliseconds, configurable as a model parameter
    Returns:
        Model
    """
    model = RandomProjectionMelspecGCCEmbedding(**model_options)
    if model_file_path != "":
        loaded_model = torch.load(model_file_path)
        if not isinstance(loaded_model, OrderedDict):
            raise TypeError(
                f"Loaded model must be a model state dict of type OrderedDict. "
                f"Received {type(loaded_model)}"
            )

        model.load_state_dict(loaded_model)

    return model


def get_timestamp_embeddings(
    audio: Tensor,
    model: torch.nn.Module,
    hop_size: Optional[float] = None,
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
            "audio input tensor must be stereo/binaural"
        )

    # Make sure the correct model type was passed in
    if not isinstance(model, RandomProjectionMelspecGCCEmbedding):
        raise ValueError(
            f"Model must be an instance of {RandomProjectionMelspecGCCEmbedding.__name__}"
        )

    # Send the model to the same device that the audio tensor is on.
    model = model.to(audio.device)

    hop_size = hop_size or model.hop_size or TIMESTAMP_HOP_SIZE

    # Split the input audio signals into frames and then flatten to create a tensor
    # of audio frames that can be batch processed.
    frames, timestamps = frame_audio(
        audio,
        frame_size=WINDOW_SIZE,
        hop_size=hop_size,
        sample_rate=RandomProjectionMelspecGCCEmbedding.sample_rate,
    )
    audio_batches, num_channels, num_frames, frame_size = frames.shape
    # frames : (audio_batches * num_frames, num_channels, frame_size)
    frames = frames.transpose(1, 2).flatten(end_dim=1)

    # We're using a DataLoader to help with batching of frames
    dataset = torch.utils.data.TensorDataset(frames)
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


def get_scene_embeddings(
    audio: Tensor,
    model: torch.nn.Module,
    hop_size: Optional[float] = None,
) -> Tensor:
    """
    This function returns a single embedding for each audio clip. In this baseline
    implementation we simply summarize the temporal embeddings from
    get_timestamp_embeddings() using torch.mean().

    Args:
        audio: n_sounds x n_channels x n_samples of audio in the range [-1, 1]. All sounds in
            a batch will be padded/trimmed to the same length.
        model: Loaded model.
        hop_size: Hop size in milliseconds.
            NOTE: Not required by the HEAR API. We add this optional parameter
            to improve the efficiency of scene embedding.

    Returns:
        - embeddings, A float32 Tensor with shape
            (n_sounds, model.scene_embedding_size).
    """
    hop_size = hop_size or model.hop_size or SCENE_HOP_SIZE
    embeddings, _ = get_timestamp_embeddings(audio, model, hop_size=hop_size)
    embeddings = torch.mean(embeddings, dim=1)
    return embeddings

