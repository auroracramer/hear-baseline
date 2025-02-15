"""
torchopenl3 model for HEAR 2021 NeurIPS competition.
"""

import functools
from typing import Tuple

import torch
import torchopenl3
from torch import Tensor

TIMESTAMP_HOP_SIZE = 50
SCENE_HOP_SIZE = 250


def load_model(
    model_file_path: str = "",
    input_repr="mel256",
    content_type="music",
    embedding_size=6144,
    center=True,
    batch_size=32,
    verbose=False,
    # Concatenate, don't mean, to get timestamp embeddings
    # You probably want a larger hop-size for this
    scene_embedding_mean=True,
    # Length of audio used for the scene embedding
    scene_embedding_audio_length_ms=4000,
) -> torch.nn.Module:
    """
    Returns a torch.nn.Module that produces embeddings for audio.

    Args:
        model_file_path: Ignored.
    Returns:
        Model
    """
    model = torchopenl3.core.load_audio_embedding_model(
        input_repr, content_type, embedding_size
    )
    if torch.cuda.is_available():
        model.cuda()
    model.eval()

    model.sample_rate = 48000
    model.num_channels = 1
    model.embedding_size = embedding_size
    model.timestamp_embedding_size = embedding_size
    if scene_embedding_mean:
        model.scene_embedding_size = embedding_size
    else:
        # center padding on start and end, so add 1
        model.scene_embedding_size = embedding_size * (
            int(scene_embedding_audio_length_ms / TIMESTAMP_HOP_SIZE) + 1
        )

    # model.center=center
    # model.batch_size=batch_size
    # model.verbose=verbose
    model.scene_embedding_mean = scene_embedding_mean
    model.scene_embedding_audio_length_ms = scene_embedding_audio_length_ms

    model.get_audio_embedding = functools.partial(
        torchopenl3.core.get_audio_embedding,
        sr=model.sample_rate,
        model=model,
        center=center,
        batch_size=batch_size,
        verbose=verbose,
    )
    return model


def get_timestamp_embeddings(
    audio: Tensor,
    model: torch.nn.Module,
    hop_size: float = TIMESTAMP_HOP_SIZE,
) -> Tuple[Tensor, Tensor]:
    """
    This function returns embeddings at regular intervals centered at timestamps. Both
    the embeddings and corresponding timestamps (in milliseconds) are returned.

    Args:
        audio: n_sounds x n_channels x n_samples of audio in the range [-1, 1].
        model: Loaded model.
        hop_size: Hop size in milliseconds.

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

    if audio.shape[1] != 1:
        raise ValueError(
            "audio input tensor must be mono"
        )

    # Remove channel dimension for mono model
    audio = audio.squeeze(1)

    # Make sure the correct model type was passed in
    if not isinstance(model, torchopenl3.models.PytorchOpenl3):
        raise ValueError(
            f"Model must be an instance of {torchopenl3.models.PytorchOpenl3.__name__}"
        )

    # Send the model to the same device that the audio tensor is on.
    # model = model.to(audio.device)

    # Put the model into eval mode, and not computing gradients while in inference.
    # Iterate over all batches and accumulate the embeddings for each frame.
    with torch.no_grad():
        # Pad by up to 1/2 frame (0.5 seconds), so that we get a
        # timestamp at the end of audio

        padded_audio = torch.nn.functional.pad(
            audio,
            (
                0,
                int(model.sample_rate / 2)
                - audio.shape[1] % int(model.sample_rate * hop_size / 1000),
            ),
            mode="constant",
            value=0,
        )
        embeddings, timestamps = model.get_audio_embedding(
            padded_audio, hop_size=hop_size / 1000
        )

    # seconds to ms
    timestamps = timestamps * 1000
    assert timestamps.shape[1] == embeddings.shape[1]

    return embeddings, timestamps


def get_scene_embeddings(
    audio: Tensor,
    model: torch.nn.Module,
) -> Tensor:
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
    if model.scene_embedding_mean:
        embeddings, _ = get_timestamp_embeddings(audio, model, hop_size=SCENE_HOP_SIZE)
        embeddings = torch.mean(embeddings, dim=1)
    else:
        # Trim or pad to, say, 4 seconds and concat the embeddings
        pad_samples = int(
            model.scene_embedding_audio_length_ms / 1000 * model.sample_rate
        )
        if audio.shape[-1] > pad_samples:
            audio = audio[..., :pad_samples]
        else:
            audio = torch.nn.functional.pad(
                audio, (0, pad_samples - audio.shape[-1]), "constant", 0
            )
        assert audio.shape[-1] == pad_samples
        embeddings, timestamps = get_timestamp_embeddings(
            audio, model, hop_size=SCENE_HOP_SIZE
        )
        embeddings = embeddings.view(embeddings.shape[0], -1)
    return embeddings
