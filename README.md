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

The model was pretrained from scratch using **31,255 hours** of speech data from the **Vaani** dataset, covering **105 languages**. It then underwent an **audio-image alignment** stage using **11 million audio-image pairs** from the **Vaani** dataset to leverage multimodal relationships for learning richer audio representations..

Finally, the model was fine-tuned on approximately **31,270 hours** of transcribed speech data spanning **63 Indian languages and dialects**, using a combination of the Vaani dataset and multiple open-source speech datasets.

## Supported Languages

The model supports **63 Indian languages and dialects**, covering all major language families spoken across the country. It includes widely spoken scheduled languages such as Assamese, Bengali, Bodo, Dogri, Gujarati, Hindi, Kannada, Konkani, Maithili, Malayalam, Manipuri, Marathi, Nepali, Odia, Punjabi, Sanskrit, Santali, Sindhi, Tamil, and Telugu.

In addition, the model supports numerous regional and low-resource languages and dialects, including Angami, Ao, Awadhi, Bajjika, Bearybashe, Bhili, Bhojpuri, Bundeli, Chakhesang, Chakma, Chhattisgarhi, Garo, Garhwali, Gondi, Halbi, Haryanvi, Idu Mishmi, Karbi, Khariboli, Khortha, Kokborok, Kurukh, Magadhi, Malvani, Marwari, Mizo, Nagamese, Nyishi, Rajasthani, Rengma, Rongmei, Sadri, Sambalpuri, Sumi, Surgujia, Surjapuri, Tagin, Tulu, and Wancho.

**Note:** Although the pretraining corpus covered 105 languages, this released model is fine-tuned for **63 Indian languages and dialects**. **Urdu and Kashmiri are not supported** in this release.


## Sample Usage
```
import sys, torch
from transformers import AutoModel

REPO = "ARTPARK-IISc/SraVaani"
DEV = "cuda" if torch.cuda.is_available() else "cpu"
model = AutoModel.from_pretrained(REPO, trust_remote_code=True).to(DEV).eval()


hyps = model.transcribe(sys.argv[1:], return_hypotheses=True)
for path, h in zip(sys.argv[1:], hyps):
    print(f"{path}\t{h.text}")

# --- lower-level alternative (explicit processor) ---
# from transformers import AutoProcessor
# import soundfile as sf   
# proc = AutoProcessor.from_pretrained(REPO, trust_remote_code=True)
# wav, sr = sf.read(path, dtype="float32")   
# inputs = proc(wav, sampling_rate=sr, return_tensors="pt").to(DEV)
# text = proc.batch_decode(model.generate(**inputs))[0]

```
## Performance
## Scheduled languages

| Language | Avg WER | CommonVoice | Fleurs | GramVaani | IndicTTS | Kathbath  | MUCS | RESPIN | Vaani  |
|---|---|---|---|---|---|---|---|---|---|
| Assamese | 19.4 | 22.6 | 18.8 | – | – | – | – | – | 16.9 |
| Bengali | 19.8 | 14.1 | 15.2 | – | 26.2 | 12.3 | – | 26.9 | 24.1 |
| Gujarati | 19.0 | – | 18.2 | – | 18.4 | 14.9 | 26.6 | – | 16.8 |
| Hindi | 12.4 | 12.7 | 10.9 | 20.93 | 9.1 | 9.1 | 14.6 | 10.8 | 10.8 |
| Kannada | 27.4 | – | 21.4 | – | 17.4 | 16.6 | – | 27.4 | 54.1 |
| Konkani | 54.7 | – | – | – | – | – | – | – | 54.7 |
| Maithili | 27.5 | – | – | – | – | – | – | 20.2 | 34.8 |
| Malayalam | 27.7 | 33.5 | 19.9 | – | 19.4 | 31.5 | – | – | 34.1 |
| Manipuri | 41.6 | – | – | – | – | – | – | – | 41.6 |
| Marathi | 19.7 | 16.6 | 18.1 | – | 14.2 | 17.8 | 22.0 | 16.1 | 33.4 |
| Nepali | 31.3 | 45.8 | 26.6 | – | – | – | – | – | 21.5 |
| Odia | 25.7 | 29.4 | 20.3 | – | 18.7 | 21.4 | 21.4 | – | 43.2 |
| Punjabi | 20.2 | 29.3 | 15.9 | – | – | 13.6 | – | – | 22.1 |
| Sanskrit | 36.4 | – | – | – | – | 36.4 | – | – | – |
| Santali | 57.3 | – | – | – | – | – | – | – | 57.3 |
| Tamil | 26.2 | 28.9 | 28.1 | – | 21.1 | 23.3 | 24.1 | – | 32.0 |
| Telugu | 25.1 | – | 23.1 | – | 20.9 | 21.3 | 26.4 | 25.4 | 33.5 |

## Non-scheduled languages

| Language | Avg WER | RESPIN | Vaani  |
|---|---|---|---|
| Angika | 31.4 | – | 31.4 |
| Ao | 57.7 | – | 57.7 |
| Awadhi | 43.8 | – | 43.8 |
| Bajjika | 34.4 | – | 34.4 |
| Bantar | 5.6 | – | 5.6 |
| Bearybashe | 77.8 | – | 77.8 |
| Bhatri | 60.3 | – | 60.3 |
| Bhili | 73.5 | – | 73.5 |
| Bhojpuri | 27.8 | 20.8 | 34.8 |
| Bundeli | 36.0 | – | 36.0 |
| Chakhesang | 75.8 | – | 75.8 |
| Chakma | 51.2 | – | 51.2 |
| Chhattisgarhi | 27.4 | 14.5 | 40.3 |
| English | 14.8 | – | 14.8 |
| Garhwali | 53.5 | – | 53.5 |
| Garo | 9.5 | – | 9.5 |
| Halbi | 51.6 | – | 51.6 |
| Haryanvi | 48.1 | – | 48.1 |
| Idu Mishmi | 64.1 | – | 64.1 |
| Jaipuri | 28.1 | – | 28.1 |
| Karbi | 63.3 | – | 63.3 |
| Khandeshi | 56.6 | – | 56.6 |
| Khariboli | 27.0 | – | 27.0 |
| Khorth/Khortha/Khorthkhotta | 41.3 | – | 41.3 |
| Kokborok | 68.2 | – | 68.2 |
| Kumaoni | 30.4 | – | 30.4 |
| Kurumali | 11.3 | – | 11.3 |
| Lambadi | 25.8 | – | 25.8 |
| Magadhi | 23.0 | 23.0 | – |
| Magadhi/Magahi | 37.8 | – | 37.8 |
| Malvani | 30.4 | – | 30.4 |
| Marwari | 39.7 | – | 39.7 |
| Mizo | 25.3 | – | 25.3 |
| Nagamese | 50.1 | – | 50.1 |
| Nagpuri/Sadri | 58.8 | – | 58.8 |
| Powari | 54.2 | – | 54.2 |
| Rajasthani | 41.8 | – | 41.8 |
| Rengma | 71.7 | – | 71.7 |
| Sambalpuri | 62.7 | – | 62.7 |
| Shekhawati | 19.1 | – | 19.1 |
| Sumi | 78.5 | – | 78.5 |
| Surgujia | 37.5 | – | 37.5 |
| Surjapuri | 69.1 | – | 69.1 |
| Tulu | 57.3 | – | 57.3 |
| Wancho | 67.6 | – | 67.6 |

## Limitations

Although the model supports 63 Indian languages and dialects, the amount of fine-tuning data varies significantly across languages. For several low-resource languages, the available training data is limited, which may result in lower recognition accuracy compared to high-resource languages.

The model **does not support Urdu or Kashmiri**.


## Feedback

We welcome feedback, suggestions, and bug reports from the community. If you encounter any issues or have questions about the model, please contact us at:

**[vaanicontact@gmail.com](mailto:vaanicontact@gmail.com)**


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