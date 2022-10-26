"""
variable-Q transform spectrogram (60 bins/octave, ~10ms step)

Based upon DCASE 2016 Task 2 baseline.
The scene embeddings will be an average, i.e. only 84 dimensions.
"""

import hearbaseline.mono.vqt as vqt
from hearbaseline.util import mono_module_to_multichannel_module


(
    load_model,
    get_timestamp_embeddings,
    get_scene_embeddings,
) = mono_module_to_multichannel_module(vqt, num_channels=2)