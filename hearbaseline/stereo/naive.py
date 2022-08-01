"""
Baseline model for HEAR 2021 NeurIPS competition.

This is simply a mel spectrogram followed by random projection.
"""

import hearbaseline.mono.naive as naive
from hearbaseline.mono.naive import (
    TIMESTAMP_HOP_SIZE, SCENE_HOP_SIZE, BATCH_SIZE
)
from hearbaseline.util import mono_module_to_multichannel_module

(
    load_model,
    get_timestamp_embeddings,
    get_scene_embeddings,
) = mono_module_to_multichannel_module(
    naive,
    num_channels=2,
    inherit_model_attrs=("n_fft",)
)