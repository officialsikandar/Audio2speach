"""HF processor: raw audio -> mel input_features, and token ids -> text (SentencePiece)."""
import os
import torch
import sentencepiece as spm
from transformers.feature_extraction_utils import BatchFeature


def _resample(wav, sr_from, sr_to):
    """Fourier-method resampling (scipy.signal.resample style), pure torch."""
    sr_from, sr_to = int(sr_from), int(sr_to)
    if sr_from == sr_to or wav.shape[-1] == 0:
        return wav
    n = wav.shape[-1]
    n_new = max(int(round(n * sr_to / sr_from)), 1)
    X = torch.fft.rfft(wav)
    Y = torch.zeros(n_new // 2 + 1, dtype=X.dtype, device=wav.device)
    k = min(X.shape[-1], Y.shape[-1])
    Y[:k] = X[:k]
    return (torch.fft.irfft(Y, n=n_new) * (n_new / n)).float()


def _normalize_per_feature(x, seq_len, constant):
    B, _, max_time = x.shape
    steps = torch.arange(max_time, device=x.device).unsqueeze(0).expand(B, max_time)
    valid = steps < seq_len.unsqueeze(1)
    denom = valid.sum(dim=1)
    mean = torch.where(valid.unsqueeze(1), x, torch.zeros_like(x)).sum(dim=2) / denom.unsqueeze(1)
    var = torch.sum(torch.where(valid.unsqueeze(1), x - mean.unsqueeze(2), torch.zeros_like(x)) ** 2,
                    dim=2) / (denom.unsqueeze(1) - 1.0)
    std = torch.sqrt(var).masked_fill(torch.sqrt(var).isnan(), 0.0) + constant
    return (x - mean.unsqueeze(2)) / std.unsqueeze(2)


class SraVaaniProcessor:
    def __init__(self, sp, window, fb, params, blank_id=3000, sample_rate=16000):
        self.sp = sp
        self.window = window
        self.fb = fb
        self.p = params
        self.blank_id = blank_id
        self.sample_rate = sample_rate

    @classmethod
    def register_for_auto_class(cls, auto_class="AutoProcessor"):
        return None

    @classmethod
    def from_pretrained(cls, path, **kwargs):
        sp = spm.SentencePieceProcessor(model_file=os.path.join(path, "tokenizer.model"))
        pp = torch.load(os.path.join(path, "preproc.pt"), map_location="cpu")
        blank = pp.get("blank_id", 3000)
        return cls(sp, pp["window"], pp["fb"], pp["params"], blank_id=blank)

    def save_pretrained(self, path, **kwargs):
        os.makedirs(path, exist_ok=True)

    @staticmethod
    def _to_2d(audio):
        if isinstance(audio, torch.Tensor):
            a = audio
        else:
            import numpy as np
            a = torch.as_tensor(np.asarray(audio), dtype=torch.float32)
        if a.dim() == 1:
            a = a.unsqueeze(0)
        return a.float()

    @torch.no_grad()
    def __call__(self, audio, sampling_rate=16000, return_tensors="pt"):
        wav = self._to_2d(audio)
        if sampling_rate != self.sample_rate:
            wav = _resample(wav, sampling_rate, self.sample_rate)
        p = self.p
        n_fft, hop, win = p["n_fft"], p["hop_length"], p["win_length"]
        wav_len = torch.tensor([wav.shape[1]] * wav.shape[0])
        seq_len = torch.div(wav_len + n_fft - n_fft, hop, rounding_mode="floor")
        tmask = torch.arange(wav.shape[1]).unsqueeze(0) < wav_len.unsqueeze(1)
        x = torch.cat((wav[:, :1], wav[:, 1:] - p["preemph"] * wav[:, :-1]), dim=1).masked_fill(~tmask, 0.0)
        spec = torch.stft(x, n_fft=n_fft, hop_length=hop, win_length=win, window=self.window,
                          center=True, pad_mode="constant", return_complex=True)
        x = torch.view_as_real(spec)
        x = torch.sqrt(x.pow(2).sum(-1)).pow(p["mag_power"])
        x = torch.matmul(self.fb.to(x.dtype), x)
        x = torch.log(x + p["log_zero_guard_value"])
        x = _normalize_per_feature(x, seq_len, p["CONSTANT"])
        max_len = x.size(-1)
        m = torch.arange(max_len).repeat(x.size(0), 1) >= seq_len.unsqueeze(1)
        x = x.masked_fill(m.unsqueeze(1), p["pad_value"])
        return BatchFeature({"input_features": x, "feature_lengths": seq_len}, tensor_type=return_tensors)

    def _clean(self, ids):
        return [int(i) for i in ids if int(i) != self.blank_id]

    def batch_decode(self, sequences, **kwargs):
        token_lists = getattr(sequences, "token_lists", None)
        if token_lists is None:
            seqs = getattr(sequences, "sequences", sequences)
            if isinstance(seqs, torch.Tensor):
                token_lists = [self._clean(row.tolist()) for row in seqs]
            else:
                token_lists = [self._clean(row) for row in seqs]
        return [self.sp.decode(t) for t in token_lists]

    def decode(self, sequence, **kwargs):
        return self.batch_decode([sequence], **kwargs)[0]
