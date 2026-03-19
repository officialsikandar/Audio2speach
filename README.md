---
license: mit
pipeline_tag: automatic-speech-recognition
language:
- hi
- kn
- ml
- te
- en
- gu
- pa
- or
- bn
- ta
- as
- sa
- ne
- mr
datasets:
- ARTPARK-IISc/Vaani
- ARTPARK-IISc/Vaani-transcription-part
---
```python

from nemo.collections.asr.models import EncDecRNNTBPEModel

# Load from HF Hub
model = EncDecRNNTBPEModel.from_pretrained(model_name="ARTPARK-IISc/Vaani-FastConformer-Multilingual")


# Path to your audio file
audioPath = "sample.wav"

# Transcribe the audio

hypotheses = model.transcribe([audioPath], return_hypotheses=True)

print("Transcription:", hypotheses[0].text)

```