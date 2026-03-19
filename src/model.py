import torch
import torch.nn as nn
from src import config


class SpectralTransformer(nn.Module):
    """
    Masked Transformer Encoder for few-shot audio signal in-painting.

    The model receives a partially observed audio clip where missing time
    steps are zeroed out. A binary mask explicitly tells the model which
    positions are known (context) and which must be reconstructed (target).

    Bidirectional self-attention allows every missing position to attend
    to all surviving context points simultaneously — regardless of their
    distance in time — making it well suited for irregular gap patterns.

    Args:
        seq_len     : number of time steps per clip (default: config.SEQ_LEN)
        feature_dim : input features per time step (default: config.FEATURE_DIM)
        d_model     : internal embedding dimension (default: config.D_MODEL)
        n_heads     : number of attention heads (default: config.N_HEADS)
        n_layers    : number of Transformer encoder layers (default: config.N_LAYERS)
        d_ff        : feed-forward hidden dimension (default: config.D_FF)
        dropout     : dropout rate on attention and FFN (default: config.DROPOUT)
        activation  : activation function — 'gelu' or 'relu' (default: config.ACTIVATION)

    Input shape  : [batch, seq_len, feature_dim]
    Output shape : [batch, seq_len]
    """

    def __init__(
        self,
        seq_len:     int   = None,
        feature_dim: int   = None,
        d_model:     int   = None,
        n_heads:     int   = None,
        n_layers:    int   = None,
        d_ff:        int   = None,
        dropout:     float = None,
        activation:  str   = None,
    ):
        super().__init__()

        # fall back to config defaults for any argument not provided
        self.seq_len     = seq_len     or config.SEQ_LEN
        self.feature_dim = feature_dim or config.FEATURE_DIM
        self.d_model     = d_model     or config.D_MODEL
        self.n_heads     = n_heads     or config.N_HEADS
        self.n_layers    = n_layers    or config.N_LAYERS
        self.d_ff        = d_ff        or config.D_FF
        self.dropout     = dropout     if dropout is not None else config.DROPOUT
        self.activation  = activation  or config.ACTIVATION

        # ── 1. Input projection ───────────────────────────────────────────
        # Projects the 2-feature input into d_model dimensions so the
        # Transformer has a rich enough space to work in.
        self.input_proj = nn.Linear(self.feature_dim, self.d_model)

        # ── 2. Positional encoding ────────────────────────────────────────
        # Learnable embeddings that tell the model the absolute position
        # of each time step. Audio frequencies depend on time order, so
        # this is essential. Initialized with small random values.
        self.pos_encoding = nn.Parameter(
            torch.randn(1, self.seq_len, self.d_model) * 0.02
        )

        # ── 3. Transformer encoder ────────────────────────────────────────
        # Each layer applies multi-head self-attention followed by a
        # position-wise feed-forward network with residual connections
        # and layer normalisation. Dropout is applied inside both
        # sub-layers, which your friend's version did not have.
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=self.n_heads,
            dim_feedforward=self.d_ff,
            dropout=self.dropout,
            activation=self.activation,
            batch_first=True,   # input shape is [batch, seq, dim]
            norm_first=False,   # post-norm (original Transformer convention)
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=self.n_layers,
            enable_nested_tensor=False,  # avoids a warning on some PyTorch versions
        )

        # ── 4. Output projection ──────────────────────────────────────────
        # Collapses the d_model representation back to a single voltage
        # prediction per time step.
        self.output_proj = nn.Linear(self.d_model, 1)

        # initialise weights sensibly
        self._init_weights()

    # ── forward pass ──────────────────────────────────────────────────────

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x : [batch, seq_len, feature_dim]  — masked input tensor

        Returns:
            [batch, seq_len]  — predicted voltage for every time step.
            Loss is computed only on target positions (mask == 0).
        """
        # project input features into embedding space
        x = self.input_proj(x)            # [batch, seq_len, d_model]

        # add positional information
        x = x + self.pos_encoding         # [batch, seq_len, d_model]

        # run through stacked Transformer encoder layers
        x = self.transformer(x)           # [batch, seq_len, d_model]

        # project each time step down to a single voltage value
        x = self.output_proj(x)           # [batch, seq_len, 1]

        return x.squeeze(-1)              # [batch, seq_len]

    # ── uncertainty estimation ────────────────────────────────────────────

    def predict_with_uncertainty(
        self,
        x:          torch.Tensor,
        n_samples:  int = 20,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Monte Carlo Dropout inference — runs the model n_samples times
        with dropout active to estimate prediction uncertainty.

        Dropout is normally disabled during eval(). Here we force it on
        so each forward pass samples a different sub-network. The spread
        across runs gives a confidence interval: wide spread = uncertain,
        narrow spread = confident.

        Args:
            x         : [batch, seq_len, feature_dim]
            n_samples : number of stochastic forward passes (default: 20)

        Returns:
            mean : [batch, seq_len]  — mean prediction across all samples
            std  : [batch, seq_len]  — std deviation (uncertainty estimate)
        """
        # keep dropout active during inference for MC sampling
        self._enable_dropout()

        with torch.no_grad():
            preds = torch.stack(
                [self.forward(x) for _ in range(n_samples)], dim=0
            )  # [n_samples, batch, seq_len]

        self.eval()  # restore normal eval behaviour

        mean = preds.mean(dim=0)   # [batch, seq_len]
        std  = preds.std(dim=0)    # [batch, seq_len]

        return mean, std

    # ── utility ───────────────────────────────────────────────────────────

    def parameter_count(self) -> dict[str, int]:
        """Returns total and trainable parameter counts."""
        total     = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {"total": total, "trainable": trainable}

    def __repr__(self) -> str:
        p = self.parameter_count()
        return (
            f"SpectralTransformer("
            f"d_model={self.d_model}, "
            f"n_heads={self.n_heads}, "
            f"n_layers={self.n_layers}, "
            f"d_ff={self.d_ff}, "
            f"dropout={self.dropout}, "
            f"params={p['trainable']:,})"
        )

    # ── private helpers ───────────────────────────────────────────────────

    def _init_weights(self) -> None:
        """
        Initialise linear layer weights with Xavier uniform and zero biases.
        This gives more stable early training than PyTorch's default init.
        """
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def _enable_dropout(self) -> None:
        """Force all dropout layers into training mode for MC sampling."""
        for module in self.modules():
            if isinstance(module, nn.Dropout):
                module.train()


# ── factory function ───────────────────────────────────────────────────────


def build_model(cfg: dict = None) -> SpectralTransformer:
    """
    Build a SpectralTransformer from a config dictionary.

    Used by the ablation notebook to instantiate each config cleanly:

        for cfg in config.ABLATION_CONFIGS:
            model = build_model(cfg)

    If cfg is None, builds the default model from config.py.

    Args:
        cfg : dict with any subset of keys:
              name, d_model, n_heads, n_layers, d_ff, dropout, activation

    Returns:
        SpectralTransformer instance
    """
    if cfg is None:
        return SpectralTransformer()

    return SpectralTransformer(
        d_model    = cfg.get("d_model",    config.D_MODEL),
        n_heads    = cfg.get("n_heads",    config.N_HEADS),
        n_layers   = cfg.get("n_layers",   config.N_LAYERS),
        d_ff       = cfg.get("d_ff",       config.D_FF),
        dropout    = cfg.get("dropout",    config.DROPOUT),
        activation = cfg.get("activation", config.ACTIVATION),
    )


# ── loss function ──────────────────────────────────────────────────────────


def masked_mse_loss(
    y_pred: torch.Tensor,
    y_true: torch.Tensor,
    mask:   torch.Tensor,
) -> torch.Tensor:
    """
    MSE loss computed exclusively on target positions (mask == 0).

    Training only on gap positions forces the model to learn the
    underlying signal physics rather than memorising context values
    it was already given.

    Args:
        y_pred : [batch, seq_len]  — model predictions
        y_true : [batch, seq_len]  — ground truth voltages
        mask   : [batch, seq_len]  — 1 = context (known), 0 = target (gap)

    Returns:
        scalar loss tensor
    """
    target_mask    = 1.0 - mask                          # 1 on gaps, 0 on context
    squared_errors = (y_pred - y_true) ** 2
    masked_errors  = squared_errors * target_mask
    loss = masked_errors.sum() / (target_mask.sum() + 1e-8)
    return loss