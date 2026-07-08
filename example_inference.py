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
