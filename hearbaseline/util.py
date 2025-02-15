"""
Utility functions for hear-kit
"""

from types import ModuleType
from typing import Callable, Iterable, Optional, Tuple

import torch
import torch.nn
import torch.nn.functional as F
from torch import Tensor
import tensorflow as tf
import numpy as np

def compute_stft(
    x: Tensor, win_length: int, hop_length: int, n_fft: int,
    pad_mode: str = "constant", center: bool = True,
    window: Optional[Tensor] = None,
):
    """
    Args:
        x: time-domain float32 tensor (N,S) or (N,ch,S)
            N: batch size
            ch: number of channels, optional
            S: number of samples
    Returns: (N,F,T,F) or (N,ch,T,F) complex64 tensor
            N: batch size
            ch: number of channels, optional
            T: time frames
            F: nfft/2+1
    """
    num_channels = x.shape[1]
    win_length = int(win_length) if win_length is not None else n_fft

    stft = torch.stack(
        [
            torch.stft(
                input=x[:, ch],
                win_length=win_length,
                hop_length=hop_length,
                n_fft=n_fft,
                center=center,
                window=(
                    window if (window is not None)
                    else torch.hann_window(win_length)
                ),
                pad_mode=pad_mode, # constant for zero padding
                return_complex=True
            )
            for ch in range(num_channels)
        ],
        dim=1
    ).transpose(-2, -1)

    return stft

def frame_audio(
    audio: Tensor, frame_size: int, hop_size: float, sample_rate: int
) -> Tuple[Tensor, Tensor]:
    """
    Slices input audio into frames that are centered and occur every
    sample_rate * hop_size samples. We round to the nearest sample.

    Args:
        audio: input audio, expects a 3d Tensor of shape:
            (n_sounds, num_channels, num_samples)
        frame_size: the number of samples each resulting frame should be
        hop_size: hop size between frames, in milliseconds
        sample_rate: sampling rate of the input audio

    Returns:
        - A Tensor of shape (n_sounds, num_channels, num_frames, frame_size)
        - A Tensor of timestamps corresponding to the frame centers with shape:
            (n_sounds, num_frames).
    """

    # Zero pad the beginning and the end of the incoming audio with half a frame number
    # of samples. This centers the audio in the middle of each frame with respect to
    # the timestamps.
    audio = F.pad(audio, (frame_size // 2, frame_size - frame_size // 2))
    num_padded_samples = audio.shape[-1]

    frame_step = hop_size / 1000.0 * sample_rate
    frame_number = 0
    frames = []
    timestamps = []
    frame_start = 0
    frame_end = frame_size
    while True:
        frames.append(audio[:, :, frame_start:frame_end])
        timestamps.append(frame_number * frame_step / sample_rate * 1000.0)

        # Increment the frame_number and break the loop if the next frame end
        # will extend past the end of the padded audio samples
        frame_number += 1
        frame_start = int(round(frame_number * frame_step))
        frame_end = frame_start + frame_size

        if not frame_end <= num_padded_samples:
            break

    # Expand out the timestamps to have shape (n_sounds, num_frames)
    timestamps_tensor = torch.tensor(timestamps, dtype=torch.float32)
    timestamps_tensor = timestamps_tensor.expand(audio.shape[0], -1)

    return torch.stack(frames, dim=2), timestamps_tensor


def mono_module_to_multichannel_module(
    module: ModuleType,
    num_channels: int,
    backend: str = "torch",
    inherit_model_attrs: Optional[Iterable[str]] = None,
) -> Tuple[
        Callable[..., torch.nn.Module],
        Callable[..., Tuple[Tensor, Tensor]],
        Callable[..., Tensor]
    ]:
    """Returns module functions for a module extracting embeddings for
        multi-channel audio by concatenating single channel embeddings for
        each channel"""

    if backend == "torch":
        Module_ = torch.nn.Module
        Tensor_ = torch.Tensor
        reshape_ = lambda x, shape: x.reshape(*shape)
        permute_ = lambda x, perm: x.permute(*perm)
    elif backend in ("tf", "keras"):
        Module_ = tf.Module if backend == "tf" else tf.keras.Model
        Tensor_ = tf.Tensor
        reshape_ = lambda x, shape: tf.reshape(x, shape)
        permute_ = lambda x, perm: tf.transpose(x, perm=perm)
    else:
        raise ValueError(
            f"Invalid backend: {backend}, must be 'torch', 'tf', or 'keras'"
        )

    class ModelWrapper(Module_):
        def __init__(self, model: Module_, num_channels: int):
            super().__init__()
            self.model = model
            self.num_channels = num_channels

        @property
        def sample_rate(self):
            return self.model.sample_rate

        @property
        def timestamp_embedding_size(self):
            return self.model.timestamp_embedding_size * self.num_channels

        @property
        def scene_embedding_size(self):
            return self.model.scene_embedding_size * self.num_channels


    # Change class name to be specific to module
    cls_name = (
        "".join(y.capitalize() for x in module.__name__.split('.') for y in x.split('_'))
        + f"{num_channels}ChannelModelWrapper"
    )
    # https://stackoverflow.com/a/54284495
    ModelWrapper.__name__ = ModelWrapper.__qualname__ = cls_name

    # Add properties for the attributes to inherit from the model
    if inherit_model_attrs:
        for attr in inherit_model_attrs:
            setattr(
                ModelWrapper,
                attr,
                property(
                    lambda self: getattr(self.model, attr)
                ),
            )

    def load_model(
        model_file_path: str = "", *args, **kwargs
    ) -> Module_:
        """
        Returns a torch.nn.Module that produces embeddings for audio.

        Args:
            model_file_path: Load model checkpoint from this file path.
        Returns:
            Model
        """
        model = module.load_model(model_file_path, *args, **kwargs)
        return ModelWrapper(model, num_channels)


    def get_timestamp_embeddings(audio: Tensor_, model: Module_, *args, **kwargs) -> Tuple[Tensor_, Tensor_]:
        """
        This function returns embeddings at regular intervals centered at timestamps. Both
        the embeddings and corresponding timestamps (in milliseconds) are returned.

        Args:
            audio: n_sounds x n_channels x n_samples of audio in the range [-1, 1].
            model: Loaded model.

        Returns:
            - Tensor_: embeddings, A float32 Tensor with shape (n_sounds, n_timestamps,
                model.timestamp_embedding_size).
            - Tensor_: timestamps, Centered timestamps in milliseconds corresponding
                to each embedding in the output. Shape: (n_sounds, n_timestamps).
        """

        # Assert audio is of correct shape
        if audio.ndim != 3:
            raise ValueError(
                "audio input tensor must be 3D with shape (n_sounds, num_channels, num_samples)"
            )

        num_sounds, _num_channels, num_samples = audio.shape
        if _num_channels != model.num_channels:
            raise ValueError(
                f"audio input tensor must be have {model.num_channels} channels, "
                f"but got {_num_channels}"
            )

        # Make sure the correct model type was passed in
        if not isinstance(model, ModelWrapper):
            raise ValueError(
                f"Model must be an instance of {ModelWrapper.__name__} "
            )


        embeddings, timestamps = module.get_timestamp_embeddings(
            # Collapse sounds and channel dimensions
            reshape_(audio, (num_sounds * model.num_channels, 1, num_samples)),
            model.model, *args, **kwargs
        )
        _, num_timestamps, embedding_size = embeddings.shape
        # Separate sound and channel dimensions
        embeddings = reshape_(embeddings, (num_sounds, model.num_channels, num_timestamps, embedding_size))
        # Move channel dimension before embedding dimension...
        embeddings = permute_(embeddings, (0, 2, 1, 3))
        # ...so that we can properly collapse the channels into the embedding dimension,
        # which should be equivalent to stacking the embeddings for each channel
        embeddings = reshape_(embeddings, (num_sounds, num_timestamps, embedding_size * model.num_channels))

        # Separate sound and channel dimensions
        timestamps = reshape_(timestamps, (num_sounds, model.num_channels, num_timestamps))
        # Timestamps should be same for all channels, so just take the first channel's timestamps
        timestamps = timestamps[:, 0, :]
        return embeddings, timestamps

    def get_scene_embeddings(audio: Tensor_, model: Module_, *args, **kwargs) -> Tensor_:
        """
        This function returns a single embedding for each audio clip.

        Args:
            audio: n_sounds x n_channels x n_samples of audio in the range [-1, 1]. All sounds in
                a batch will be padded/trimmed to the same length.
            model: Loaded model.

        Returns:
            - embeddings, A float32 Tensor_ with shape
                (n_sounds, model.scene_embedding_size).
        """
        num_sounds, _num_channels, num_samples = audio.shape
        if _num_channels != model.num_channels:
            raise ValueError(
                f"audio input tensor must be have {model.num_channels} channels, "
                f"but got {_num_channels}"
            )

        # Make sure the correct model type was passed in
        if not isinstance(model, ModelWrapper):
            raise ValueError(
                f"Model must be an instance of {ModelWrapper.__name__} "
            )

        embeddings = module.get_scene_embeddings(
            # Collapse sounds and channel dimensions
            reshape_(audio, (num_sounds * _num_channels, 1, num_samples)),
            model.model, *args, **kwargs
        )
        _, embedding_size = embeddings.shape
        # Reshape so embeddings for each channel are stacked
        embeddings = reshape_(embeddings, (num_sounds, model.num_channels * embedding_size))

        return embeddings

    return load_model, get_timestamp_embeddings, get_scene_embeddings
