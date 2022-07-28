"""
Utility functions for hear-kit
"""

from types import ModuleType
from typing import Callable, Tuple

import torch
import torch.nn
import torch.nn.functional as F
from torch import Tensor


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

    # Expand out the timestamps to have shape (n_sounds, num_channels, num_frames)
    timestamps_tensor = torch.tensor(timestamps, dtype=torch.float32)
    timestamps_tensor = timestamps_tensor.expand(audio.shape[0], audio.shape[1], -1)

    return torch.stack(frames, dim=2), timestamps_tensor


def mono_module_to_multichannel_module(module: ModuleType, num_channels: int):
    """Returns module functions for a module extracting embeddings for
        multi-channel audio by concatenating single channel embeddings for
        each channel"""

    class ModelWrapper(torch.nn.Module):
        def __init__(self, model: torch.nn.Module, num_channels: int):
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

        def forward(self, x: Tensor):
            # I don't think this is really event necessary
            self.model(x)

    def load_model(
        model_file_path: str = "", *args, **kwargs
    ) -> torch.nn.Module:
        """
        Returns a torch.nn.Module that produces embeddings for audio.

        Args:
            model_file_path: Load model checkpoint from this file path. For this baseline,
                if no path is provided then the default random init weights for the
                linear projection layer will be used.
        Returns:
            Model
        """
        model = module.load_model(model_file_path, *args, **kwargs)
        return ModelWrapper(model, num_channels)


    def get_timestamp_embeddings(audio: Tensor, model: ModelWrapper, *args, **kwargs) -> Tuple[Tensor, Tensor]:
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

        num_sounds, _num_channels, num_samples = audio.shape
        if _num_channels != model.num_channels:
            raise ValueError(
                f"audio input tensor must be have {model.num_channels} channels, "
                f"but got {_num_channels}"
            )

        embeddings, timestamps = module.get_timestamp_embeddings(
            # Collapse sounds and channel dimensions
            audio.reshape(num_sounds * _num_channels, 1, num_samples), model.model, *args, **kwargs
        )
        _, num_timestamps, embedding_size = embeddings.shape
        # Separate sound and channel dimensions
        embeddings = embeddings.reshape(num_sounds, model.num_channels, num_timestamps, embedding_size)
        # Move channel dimension to the end...
        embeddings = embeddings.permute(0, 2, 3, 1)
        # ...so that we can properly collapse the channels into the embedding dimension,
        # which should be equivalent to concatenating the embeddings for each channel
        embeddings = embeddings.reshape(num_sounds, num_timestamps, embedding_size * model.num_channels)

        # Separate sound and channel dimensions
        timestamps = timestamps.reshape(num_sounds, model.num_channels, num_timestamps)
        # Timestamps should be same for all channels, so just take the first channel's timestamps
        timestamps = timestamps[:, 0, :]
        return embeddings, timestamps

    def get_scene_embeddings(audio: Tensor, model: ModelWrapper, *args, **kwargs) -> Tuple[Tensor, Tensor]:
        """
        This function returns a single embedding for each audio clip. In this baseline
        implementation we simply summarize the temporal embeddings from
        get_timestamp_embeddings() using torch.mean().

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
                f"audio input tensor must be have {model.num_channels} channels, "
                f"but got {_num_channels}"
            )

        embeddings = module.get_scene_embeddings(
            # Collapse sounds and channel dimensions
            audio.reshape(num_sounds * _num_channels, 1, num_samples), model.model, *args, **kwargs
        )
        _, embedding_size = embeddings.shape
        # Separate sound and channel dimensions
        embeddings = embeddings.reshape(num_sounds, model.num_channels, embedding_size)
        # Move channel dimension to the end...
        embeddings = embeddings.permute(0, 2, 1)
        # ...so that we can properly collapse the channels into the embedding dimension,
        # which should be equivalent to concatenating the embeddings for each channel
        embeddings = embeddings.reshape(num_sounds, embedding_size * model.num_channels)

        return embeddings

    return load_model, get_timestamp_embeddings, get_scene_embeddings
