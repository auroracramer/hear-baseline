"""
torchopenl3 model for HEAR 2021 NeurIPS competition.
"""

import hearbaseline.mono.torchopenl3 as torchopenl3
from hearbaseline.mono.torchopenl3 import (TIMESTAMP_HOP_SIZE, SCENE_HOP_SIZE)
from hearbaseline.util import mono_module_to_multichannel_module


(
    load_model,
    get_timestamp_embeddings,
    get_scene_embeddings,
) = mono_module_to_multichannel_module(torchopenl3, num_channels=2)