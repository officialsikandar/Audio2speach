---
license: mit
pipeline_tag: automatic-speech-recognition
---
```python

from nemo.collections.asr.models import EncDecRNNTBPEModel

# Load from HF Hub
model = EncDecRNNTBPEModel.from_pretrained(model_name="ARTPARK-IISc/Vaani-FastConformer-Hindi")


# Path to your audio file
audioPath = "sample.wav"

# Transcribe the audio

hypotheses = model.transcribe([audioPath], return_hypotheses=True)

print("Transcription:", hypotheses[0].text)

```
