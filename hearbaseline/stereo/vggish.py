"""
vggish model for HEAR 2021 NeurIPS competition.
"""

import hearbaseline.mono.vggish as vggish
from hearbaseline.util import mono_module_to_multichannel_module


(
    load_model,
    get_timestamp_embeddings,
    get_scene_embeddings,
) = mono_module_to_multichannel_module(vggish, num_channels=2)