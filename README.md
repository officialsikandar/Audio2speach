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
# Citation

If you use this model, please cite the following:
```
@misc{pulikodan2026vaanicapturinglanguagelandscape,
      title={VAANI: Capturing the language landscape for an inclusive digital India}, 
      author={Sujith Pulikodan and Abhayjeet Singh and Agneedh Basu and Nihar Desai and Pavan Kumar J and Pranav D Bhat and Raghu Dharmaraju and Ritika Gupta and Sathvik Udupa and Saurabh Kumar and Sumit Sharma and Vaibhav Vishwakarma and Visruth Sanka and Dinesh Tewari and Harsh Dhand and Amrita Kamat and Sukhwinder Singh and Shikhar Vashishth and Partha Talukdar and Raj Acharya and Prasanta Kumar Ghosh},
      year={2026},
      eprint={2603.28714},
      archivePrefix={arXiv},
      primaryClass={eess.AS},
      url={https://arxiv.org/abs/2603.28714}, 
}
```