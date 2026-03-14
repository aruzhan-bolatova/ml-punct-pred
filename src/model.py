"""
DeepPunctuation: Transformer + BiLSTM for punctuation restoration.
Reference: https://github.com/habib-wahid/punctuation-restoration

Improvements: Dropout, LayerNorm, optional CRF layer.
"""

import torch
import torch.nn as nn
try:
    from .config import PUNCTUATION_DICT, MODELS, MODEL_HF_IDS
except ImportError:
    from config import PUNCTUATION_DICT, MODELS, MODEL_HF_IDS

try:
    from torchcrf import CRF
except ImportError:
    CRF = None


class DeepPunctuation(nn.Module):
    """BERT/RoBERTa + BiLSTM + residual + LayerNorm + MLP classifier."""

    def __init__(
        self,
        pretrained_model: str,
        freeze_bert: bool = False,
        lstm_dim: int = -1,
        dropout: float = 0.1,
        lstm_layers: int = 2,
    ):
        super().__init__()
        model_id = MODEL_HF_IDS.get(pretrained_model, pretrained_model)
        model_cls, _, bert_dim = MODELS[pretrained_model][0], MODELS[pretrained_model][1], MODELS[pretrained_model][2]

        self.bert_layer = model_cls.from_pretrained(model_id)
        if freeze_bert:
            for p in self.bert_layer.parameters():
                p.requires_grad = False

        hidden_size = bert_dim if lstm_dim == -1 else lstm_dim
        lstm_output_dim = hidden_size * 2  # bidirectional
        self.lstm = nn.LSTM(
            input_size=bert_dim,
            hidden_size=hidden_size,
            num_layers=lstm_layers,
            bidirectional=True,
            dropout=dropout if lstm_layers > 1 else 0,
        )
        self.residual_proj = nn.Linear(bert_dim, lstm_output_dim)
        self.layer_norm = nn.LayerNorm(lstm_output_dim)
        self.dropout = nn.Dropout(dropout)
        self.num_labels = len(PUNCTUATION_DICT)
        self.classifier = nn.Sequential(
            nn.Linear(lstm_output_dim, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, self.num_labels),
        )

    def forward(self, x, attn_masks):
        if len(x.shape) == 1:
            x = x.unsqueeze(0)
        bert_out = self.bert_layer(x, attention_mask=attn_masks)[0]
        lstm_out, _ = self.lstm(bert_out.transpose(0, 1))
        lstm_out = lstm_out.transpose(0, 1)
        x = self.layer_norm(lstm_out + self.residual_proj(bert_out))
        x = self.dropout(x)
        return self.classifier(x)


class DeepPunctuationCRF(nn.Module):
    """DeepPunctuation + CRF for sequence-consistent predictions."""

    def __init__(
        self,
        pretrained_model: str,
        freeze_bert: bool = False,
        lstm_dim: int = -1,
        dropout: float = 0.1,
        lstm_layers: int = 2,
    ):
        super().__init__()
        if CRF is None:
            raise ImportError("Install pytorch-crf: pip install pytorch-crf")
        self.bert_lstm = DeepPunctuation(
            pretrained_model,
            freeze_bert=freeze_bert,
            lstm_dim=lstm_dim,
            dropout=dropout,
            lstm_layers=lstm_layers,
        )
        self.num_labels = self.bert_lstm.num_labels
        self.crf = CRF(self.num_labels, batch_first=True)

    def log_likelihood(self, x, attn_masks, labels):
        """Training loss: negative log-likelihood of the tag sequence."""
        emissions = self.bert_lstm(x, attn_masks)
        mask = attn_masks.bool()
        return -self.crf(emissions, labels, mask=mask, reduction="token_mean")

    def decode(self, x, attn_masks):
        """Inference: Viterbi decode best tag sequence. Returns (B, N) tensor."""
        emissions = self.bert_lstm(x, attn_masks)
        mask = attn_masks.bool()
        decoded = self.crf.decode(emissions, mask=mask)  # list of lists
        # Convert to (B, N) tensor for consistency with non-CRF path
        B, N = x.shape[0], x.shape[1]
        out = torch.zeros(B, N, dtype=torch.long, device=x.device)
        for b in range(B):
            tags = decoded[b]
            k = 0
            for i in range(N):
                if mask[b, i]:
                    out[b, i] = tags[k]
                    k += 1
        return out

    def forward(self, x, attn_masks):
        """Forward returns emissions (for compatibility); use decode() for predictions."""
        return self.bert_lstm(x, attn_masks)
