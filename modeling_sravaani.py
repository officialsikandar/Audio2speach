"""
HF `transformers`-compatible inference wrapper for the SraVaani TDT (RNNT)
ASR model. Wraps the exported TorchScript encoder + decoder_joint graphs.

Runtime deps: torch, transformers, sentencepiece. **No nemo_toolkit.**
"""
import os
import torch
from transformers import PreTrainedModel
from transformers.modeling_outputs import ModelOutput
from dataclasses import dataclass
from typing import Optional, List

from .configuration_sravaani import SraVaaniTDTConfig


def _read_audio(path):
    """Load audio to a mono float32 tensor + sample rate, without torchaudio.

    Prefers soundfile (wav/flac/ogg) if installed, else the stdlib `wave` module
    (PCM WAV, 8/16/32-bit)."""
    try:
        import soundfile as sf
    except ImportError:
        sf = None
    if sf is not None:
        data, sr = sf.read(str(path), dtype="float32", always_2d=True)  # (frames, ch)
        return torch.from_numpy(data).mean(dim=1).float(), int(sr)
    import wave
    import numpy as np
    with wave.open(str(path), "rb") as w:
        sr, n, ch, sw = w.getframerate(), w.getnframes(), w.getnchannels(), w.getsampwidth()
        raw = w.readframes(n)
    if sw == 2:
        a = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    elif sw == 1:
        a = (np.frombuffer(raw, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
    elif sw == 4:
        a = np.frombuffer(raw, dtype="<i4").astype(np.float32) / 2147483648.0
    else:
        raise ValueError("unsupported WAV sample width: %d bytes (install soundfile)" % sw)
    if ch > 1:
        a = a.reshape(-1, ch).mean(axis=1)
    return torch.from_numpy(a.copy()).float(), int(sr)


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


@dataclass
class ASRGreedyOutput(ModelOutput):
    sequences: Optional[torch.LongTensor] = None
    token_lists: Optional[List[List[int]]] = None
    frame_lists: Optional[List[List[int]]] = None  # encoder frame index per emitted token


@dataclass
class Hypothesis:
    """Minimal NeMo-style hypothesis returned by transcribe(return_hypotheses=True)."""
    text: str
    y_sequence: List[int]
    score: float = 0.0
    timestamp: Optional[dict] = None  # {"word": [...], "token": [...]} when timestamps=True


class SraVaaniTDTModel(PreTrainedModel):
    """TDT/RNNT greedy ASR model wrapped for HF transformers (load via AutoModel)."""

    config_class = SraVaaniTDTConfig
    main_input_name = "input_features"

    def __init__(self, config: SraVaaniTDTConfig):
        super().__init__(config)
        self._anchor = torch.nn.Parameter(torch.zeros(1), requires_grad=False)
        self._artifacts_dir = getattr(config, "_name_or_path", ".") or "."
        self.encoder = None
        self.decoder_joint = None
        self._loaded = False
        self._io_dtype = torch.float32
        self._req_dtype = None  # user-requested compute dtype (torch_dtype / .half())
        self.post_init()

    def _ensure_loaded(self, device=None):
        if self._loaded:
            return
        d = self._artifacts_dir
        dev = device or self._anchor.device
        on_cpu = (torch.device(dev).type == "cpu")
        # The shipped graphs are fp16 on disk (half the size) but exported *unfrozen*,
        # so weights are real parameters. CPU has no fp16 conv kernel, so on CPU we
        # upcast to fp32 (lossless widening). On GPU we keep fp16 only if the user asked
        # for it (torch_dtype=float16 or .half()); otherwise we upcast for exact fp32
        # numerics. A legacy frozen fp32 graph (no parameters) is used as-is.
        want_fp16 = (self._req_dtype == torch.float16) or (self._anchor.dtype == torch.float16)

        def cast(mod):
            params = list(mod.parameters())
            pdtype = params[0].dtype if params else torch.float32
            target = torch.float16 if (want_fp16 and not on_cpu and params) else torch.float32
            if params and pdtype != target:
                mod = mod.half() if target == torch.float16 else mod.float()
            return mod, target

        def load(*names):
            path = next((os.path.join(d, n) for n in names if os.path.exists(os.path.join(d, n))),
                        os.path.join(d, names[0]))
            return cast(torch.jit.load(path, map_location=dev).eval())

        # Preferred: a single combined graph exposing .encoder and .decoder_joint
        # submodules. Fall back to the legacy two-file layout if it isn't present.
        combined = next((os.path.join(d, n) for n in ("model-asr.fp16.ts", "model-asr.ts")
                         if os.path.exists(os.path.join(d, n))), None)
        if combined is not None:
            mod, io = cast(torch.jit.load(combined, map_location=dev).eval())
            self.encoder, self.decoder_joint = mod.encoder, mod.decoder_joint
        else:
            self.encoder, io = load("encoder-asr.fp16.ts", "encoder-asr.ts")
            self.decoder_joint, _ = load("decoder_joint-asr.fp16.ts", "decoder_joint-asr.ts")
        self._io_dtype = io
        pp = torch.load(os.path.join(d, "preproc.pt"), map_location="cpu")
        self._p = pp["params"]
        self.register_buffer("_window", pp["window"].to(dev), persistent=False)
        self.register_buffer("_fb", pp["fb"].to(dev), persistent=False)
        self._loaded = True

    @classmethod
    def from_pretrained(cls, path, *args, **kwargs):
        config = kwargs.pop("config", None) or SraVaaniTDTConfig.from_pretrained(path)
        config._name_or_path = path
        model = cls(config)
        model._artifacts_dir = path
        dtype = kwargs.get("torch_dtype", None)
        if isinstance(dtype, str):
            dtype = None if dtype == "auto" else getattr(torch, dtype, None)
        if dtype in (torch.float16, torch.float32):
            model._req_dtype = dtype
            if dtype == torch.float16:
                model = model.half()
        return model

    @torch.no_grad()
    def extract_features(self, wav, wav_len):
        self._ensure_loaded()
        p = self._p
        dev = self._anchor.device
        n_fft, hop, win = p["n_fft"], p["hop_length"], p["win_length"]
        wav = wav.to(dev).float()
        wav_len = wav_len.to(dev)
        seq_len = torch.div(wav_len + n_fft - n_fft, hop, rounding_mode="floor")
        tmask = torch.arange(wav.shape[1], device=dev).unsqueeze(0) < wav_len.unsqueeze(1)
        x = torch.cat((wav[:, :1], wav[:, 1:] - p["preemph"] * wav[:, :-1]), dim=1).masked_fill(~tmask, 0.0)
        spec = torch.stft(x, n_fft=n_fft, hop_length=hop, win_length=win, window=self._window.float(),
                          center=True, pad_mode="constant", return_complex=True)
        x = torch.view_as_real(spec)
        x = torch.sqrt(x.pow(2).sum(-1)).pow(p["mag_power"])
        x = torch.matmul(self._fb.float(), x)
        x = torch.log(x + p["log_zero_guard_value"])
        x = _normalize_per_feature(x, seq_len, p["CONSTANT"])
        max_len = x.size(-1)
        m = torch.arange(max_len, device=dev).repeat(x.size(0), 1) >= seq_len.unsqueeze(1)
        return x.masked_fill(m.unsqueeze(1), p["pad_value"]), seq_len

    @torch.no_grad()
    def forward(self, input_features, feature_lengths=None, **kwargs):
        self._ensure_loaded()
        if feature_lengths is None:
            feature_lengths = torch.full((input_features.size(0),), input_features.size(-1),
                                         dtype=torch.long, device=input_features.device)
        feats = input_features.to(self._anchor.device).to(self._io_dtype)
        enc, enc_len = self.encoder(feats, feature_lengths.to(self._anchor.device))
        return ModelOutput(last_hidden_state=enc, encoder_lengths=enc_len)

    @torch.no_grad()
    def _greedy_one(self, enc_out, T):
        cfg, dev = self.config, self._anchor.device
        nd, blank, durs = cfg.num_durations, cfg.blank_id, cfg.durations
        h = torch.zeros(cfg.pred_rnn_layers, 1, cfg.pred_hidden, device=dev, dtype=self._io_dtype)
        c = torch.zeros(cfg.pred_rnn_layers, 1, cfg.pred_hidden, device=dev, dtype=self._io_dtype)
        last, toks, frames = blank, [], []
        tlen = torch.ones(1, dtype=torch.int32, device=dev)
        t = 0
        while t < T:
            f = enc_out.narrow(2, t, 1)
            added, need = 0, True
            while need and added < cfg.max_symbols:
                tgt = torch.tensor([[last]], dtype=torch.int32, device=dev)
                logits, _, h2, c2 = self.decoder_joint(f, tgt, tlen, h, c)
                logits = logits[0, 0, 0]
                k = int(logits[:-nd].argmax().item())
                skip = durs[int(logits[-nd:].argmax().item())]
                if k != blank:
                    toks.append(k); frames.append(t); h, c, last = h2, c2, k
                added += 1; t += skip; need = (skip == 0)
            if added == cfg.max_symbols:
                t += 1
        return toks, frames

    @torch.no_grad()
    def generate(self, input_features=None, feature_lengths=None, **kwargs):
        out = self.forward(input_features, feature_lengths)
        enc, enc_len = out.last_hidden_state, out.encoder_lengths
        decoded = [self._greedy_one(enc[i:i + 1], int(enc_len[i].item())) for i in range(enc.size(0))]
        token_lists = [d[0] for d in decoded]
        frame_lists = [d[1] for d in decoded]
        maxlen = max((len(t) for t in token_lists), default=0)
        pad = self.config.blank_id
        seqs = torch.full((len(token_lists), max(maxlen, 1)), pad, dtype=torch.long)
        for i, t in enumerate(token_lists):
            if t:
                seqs[i, :len(t)] = torch.tensor(t, dtype=torch.long)
        return ASRGreedyOutput(sequences=seqs, token_lists=token_lists, frame_lists=frame_lists)

    def _get_tokenizer(self):
        if getattr(self, "_sp", None) is None:
            import sentencepiece as spm
            self._sp = spm.SentencePieceProcessor(
                model_file=os.path.join(self._artifacts_dir, "tokenizer.model"))
        return self._sp

    def _frame_seconds(self):
        """Seconds of audio per encoder output frame (subsampling * hop / sample_rate)."""
        sub = int(getattr(self.config, "subsampling_factor", 8))
        return sub * float(self._p["hop_length"]) / float(self._p["sample_rate"])

    def _make_timestamps(self, sp, toks, frames, fd, total_frames):
        """Build token- and word-level timestamps from emission frames.

        Word boundaries follow SentencePiece '▁' markers. A word/token ends where
        the next one starts (last one ends at total_frames)."""
        toks = [int(x) for x in toks]
        tok_ts = []
        for i, (tok, fr) in enumerate(zip(toks, frames)):
            nxt = frames[i + 1] if i + 1 < len(frames) else total_frames
            tok_ts.append({"token": tok, "piece": sp.id_to_piece(tok),
                           "start": round(fr * fd, 3), "end": round(nxt * fd, 3)})
        groups = []  # [start_frame, [token_ids]] per word
        for tok, fr in zip(toks, frames):
            if sp.id_to_piece(tok).startswith("▁") or not groups:
                groups.append([fr, [tok]])
            else:
                groups[-1][1].append(tok)
        word_ts = []
        for gi, (sf, gtoks) in enumerate(groups):
            end = groups[gi + 1][0] if gi + 1 < len(groups) else total_frames
            word_ts.append({"word": sp.decode(gtoks),
                            "start": round(sf * fd, 3), "end": round(end * fd, 3)})
        return {"word": word_ts, "token": tok_ts}

    @torch.no_grad()
    def transcribe(self, audio, batch_size=8, return_hypotheses=False, timestamps=False, **kwargs):
        """NeMo-style convenience API.

        audio: a wav path, a 1-D waveform, or a list of either.
        Returns a list of strings, or a list of Hypothesis when return_hypotheses=True
        or timestamps=True. With timestamps=True each Hypothesis carries .timestamp =
        {"word": [{word,start,end}...], "token": [{token,piece,start,end}...]} (seconds).
        Requires sentencepiece (+ soundfile OR stdlib wave for file paths).

            hyps = model.transcribe(["sample.wav"], return_hypotheses=True, timestamps=True)
            print(hyps[0].text)
            for w in hyps[0].timestamp["word"]:
                print(w["start"], w["end"], w["word"])
        """
        self._ensure_loaded()
        sp = self._get_tokenizer()
        target_sr = int(self.config.sample_rate)
        pad_value = float(self._p["pad_value"])
        sub = int(getattr(self.config, "subsampling_factor", 8))
        fd = self._frame_seconds()
        want_hyp = return_hypotheses or timestamps
        items = [audio] if isinstance(audio, (str, bytes, os.PathLike)) else list(audio)

        results = []
        for s in range(0, len(items), batch_size):
            feats, lens = [], []
            for a in items[s:s + batch_size]:
                if isinstance(a, (str, bytes, os.PathLike)):
                    wav, sr = _read_audio(a)
                    if sr != target_sr:
                        wav = _resample(wav, sr, target_sr)
                else:
                    wav = torch.as_tensor(a, dtype=torch.float32).reshape(-1)
                f, sl = self.extract_features(wav.unsqueeze(0), torch.tensor([wav.shape[0]]))
                feats.append(f[0]); lens.append(int(sl[0]))
            dev = feats[0].device
            Fdim = feats[0].shape[0]
            Tmax = max(t.shape[1] for t in feats)
            x = torch.full((len(feats), Fdim, Tmax), pad_value, dtype=feats[0].dtype, device=dev)
            for i, t in enumerate(feats):
                x[i, :, :t.shape[1]] = t
            flen = torch.tensor(lens, dtype=torch.long, device=dev)
            out = self.generate(input_features=x, feature_lengths=flen)
            for bi, tl in enumerate(out.token_lists):
                text = sp.decode(tl)
                if not want_hyp:
                    results.append(text); continue
                hyp = Hypothesis(text=text, y_sequence=tl)
                if timestamps:
                    total_frames = -(-lens[bi] // sub)  # ceil(feature_len / subsampling)
                    hyp.timestamp = self._make_timestamps(sp, tl, out.frame_lists[bi], fd, total_frames)
                results.append(hyp)
        return results
