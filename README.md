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

## Model Overview

This is a **~430 million parameter FastConformer-based Automatic Speech Recognition (ASR)** model with a **hybrid TDT-CTC decoder**. The model is quantized to **FP16**, resulting in a total model size of approximately **900 MB**.

The model was pretrained from scratch using **31,255 hours** of speech data from the **Vaani** dataset, covering **105 languages**. It then underwent an **audio-image alignment** stage using **11 million audio-image pairs** from the Vaani dataset to improve multimodal representations.

Finally, the model was fine-tuned on approximately **31,270 hours** of transcribed speech data spanning **63 Indian languages and dialects**, using a combination of the Vaani dataset and multiple open-source speech datasets.

## Supported Languages

The model supports **63 Indian languages and dialects**, covering all major language families spoken across the country. It includes widely spoken scheduled languages such as Assamese, Bengali, Bodo, Dogri, Gujarati, Hindi, Kannada, Konkani, Maithili, Malayalam, Manipuri, Marathi, Nepali, Odia, Punjabi, Sanskrit, Santali, Sindhi, Tamil, and Telugu.

In addition, the model supports numerous regional and low-resource languages and dialects, including Angami, Ao, Awadhi, Bajjika, Bearybashe, Bhili, Bhojpuri, Bundeli, Chakhesang, Chakma, Chhattisgarhi, Garo, Garhwali, Gondi, Halbi, Haryanvi, Idu Mishmi, Karbi, Khariboli, Khortha, Kokborok, Kurukh, Magadhi, Malvani, Marwari, Mizo, Nagamese, Nyishi, Rajasthani, Rengma, Rongmei, Sadri, Sambalpuri, Sumi, Surgujia, Surjapuri, Tagin, Tulu, and Wancho.

**Note:** Although the pretraining corpus covered 105 languages, this released model is fine-tuned for **63 Indian languages and dialects**. **Urdu and Kashmiri are not supported** in this release.


## Sample Usage
```
import sys, torch
from transformers import AutoModel

REPO = "."
DEV = "cuda" if torch.cuda.is_available() else "cpu"
model = AutoModel.from_pretrained(REPO, trust_remote_code=True).to(DEV).eval()

# NeMo-style convenience API (needs sentencepiece; soundfile or stdlib wave for files):
hyps = model.transcribe(sys.argv[1:], return_hypotheses=True)
for path, h in zip(sys.argv[1:], hyps):
    print(f"{path}\t{h.text}")

# --- lower-level alternative (explicit processor) ---
# from transformers import AutoProcessor
# import soundfile as sf   # or: import wave (stdlib) for PCM WAV
# proc = AutoProcessor.from_pretrained(REPO, trust_remote_code=True)
# wav, sr = sf.read(path, dtype="float32")   # average channels if stereo
# inputs = proc(wav, sampling_rate=sr, return_tensors="pt").to(DEV)
# text = proc.batch_decode(model.generate(**inputs))[0]

```

## Limitations

Although the model supports 63 Indian languages and dialects, the amount of fine-tuning data varies significantly across languages. For several low-resource languages, the available training data is limited, which may result in lower recognition accuracy compared to high-resource languages.

The model **does not support Urdu or Kashmiri**.


## Feedback

We welcome feedback, suggestions, and bug reports from the community. If you encounter any issues or have questions about the model, please contact us at:

**[vaanicontact@gmail.com](mailto:vaanicontact@gmail.com)**

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