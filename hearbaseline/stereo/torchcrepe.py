"""
torchcrepe model for HEAR 2021 NeurIPS competition.
"""

import hearbaseline.mono.torchcrepe as torchcrepe
from hearbaseline.mono.torchcrepe import (
    SAMPLE_RATE, TIMESTAMP_HOP_SIZE, SCENE_HOP_SIZE,
    TIMESTAMP_HOP_SIZE_SAMPLES, SCENE_HOP_SIZE_SAMPLES,
)
from hearbaseline.util import mono_module_to_multichannel_module


(
    load_model,
    get_timestamp_embeddings,
    get_scene_embeddings,
) = mono_module_to_multichannel_module(torchcrepe, num_channels=2)