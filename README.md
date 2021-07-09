![HEAR2021](https://neuralaudio.ai/assets/img/hear-header-sponsor.jpg)
# HEAR 2021 Baseline

A simple DSP-based audio embedding consisting of a Mel-frequency spectrogram followed
by a random projection. Serves as the baseline model for the HEAR 2021 and implements
the [common API](https://neuralaudio.ai/hear2021-holistic-evaluation-of-audio-representations.html#common-api)
required by the competition evaluation.

For full details on the HEAR 2021 NeurIPS competition and for information on how to
participate, please visit the
[competition website.](https://neuralaudio.ai/hear2021-holistic-evaluation-of-audio-representations.html)

### Installation

**Method 1: pypi**
```python
pip install hearbaseline
```

**Method 2: pip local source tree**

This is the same method that will be used to by competition organizers when installing
submissions to HEAR 2021.
```python
git clone https://github.com/neuralaudio/hear-baseline.git
python3 -m pip install ./hear-baseline
```

### Usage

Audio embeddings can be computed using one of two methods: 1)
`get_scene_embeddings`, or 2) `get_timestamp_embeddings`.

`get_scene_embeddings` accepts a batch of audio clips and produces a single embedding
for each audio clip. This can be computed like so:
```python
import torch
import hearbaseline

# Load model with weights - located in the root directory of this repo
model = hearbaseline.load_model("./baseline_weights.pt")

# Create a batch of 2 white noise clips that are 2-seconds long
# and compute scene embeddings for each clip
audio = torch.rand((2, model.sample_rate * 2))
embeddings = hearbaseline.get_scene_embeddings(audio, model)
```

The `get_timestamp_embeddings` method works exactly the same but returns an array
of embeddings computed every 25ms over the duration of the input audio. An array
of timestamps corresponding to each embedding is also returned.

See the [common API](https://neuralaudio.ai/hear2021-holistic-evaluation-of-audio-representations.html#common-api)
for more details.
