"""
wav2vec2 model for HEAR 2021 NeurIPS competition.

Adapted from
https://colab.research.google.com/drive/17Hu1pxqhfMisjkSgmM2CnZxfqDyn2hSY?usp=sharing
"""

import hearbaseline.mono.wav2vec2 as wav2vec2
from hearbaseline.mono.wav2vec2 import (TIMESTAMP_HOP_SIZE, SCENE_HOP_SIZE)
from hearbaseline.util import mono_module_to_multichannel_module


(
    load_model,
    get_timestamp_embeddings,
    get_scene_embeddings,
) = mono_module_to_multichannel_module(wav2vec2, num_channels=2)