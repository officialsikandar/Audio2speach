from transformers import PretrainedConfig


class SraVaaniTDTConfig(PretrainedConfig):
    """Config for the SraVaani TDT (RNNT) ASR model wrapped for HF transformers."""

    model_type = "sravaani_tdt"

    def __init__(
        self,
        vocab_size=3000,
        blank_id=3000,
        durations=(0, 1, 2, 3, 4),
        num_durations=5,
        pred_hidden=640,
        pred_rnn_layers=1,
        max_symbols=10,
        enc_d_model=1024,
        feat_in=128,
        sample_rate=16000,
        subsampling_factor=8,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.blank_id = blank_id
        self.durations = list(durations)
        self.num_durations = num_durations
        self.pred_hidden = pred_hidden
        self.pred_rnn_layers = pred_rnn_layers
        self.max_symbols = max_symbols
        self.enc_d_model = enc_d_model
        self.feat_in = feat_in
        self.sample_rate = sample_rate
        self.subsampling_factor = subsampling_factor
        super().__init__(**kwargs)
