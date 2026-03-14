"""
DeepPunctuation: Transformer + BiLSTM for punctuation restoration.
Reference: https://github.com/habib-wahid/punctuation-restoration
"""

import torch
import torch.nn as nn
try:
    from .config import PUNCTUATION_DICT, MODELS, MODEL_HF_IDS
except ImportError:
    from config import PUNCTUATION_DICT, MODELS, MODEL_HF_IDS


class DeepPunctuation(nn.Module):
    """BERT/RoBERTa + BiLSTM + Linear classifier."""

    def __init__(
        self,
        pretrained_model: str,
        freeze_bert: bool = False,
        lstm_dim: int = -1,
    ):
        super().__init__()
        model_id = MODEL_HF_IDS.get(pretrained_model, pretrained_model)
        model_cls, _, bert_dim = MODELS[pretrained_model][0], MODELS[pretrained_model][1], MODELS[pretrained_model][2]

        self.bert_layer = model_cls.from_pretrained(model_id)
        if freeze_bert:
            for p in self.bert_layer.parameters():
                p.requires_grad = False

        hidden_size = bert_dim if lstm_dim == -1 else lstm_dim
        self.lstm = nn.LSTM(
            input_size=bert_dim,
            hidden_size=hidden_size,
            num_layers=1,
            bidirectional=True,
        )
        self.num_labels = len(PUNCTUATION_DICT)
        self.linear = nn.Linear(hidden_size * 2, self.num_labels)

    def forward(self, x, attn_masks):
        if len(x.shape) == 1:
            x = x.unsqueeze(0)
        # (B, N) -> (B, N, E)
        x = self.bert_layer(x, attention_mask=attn_masks)[0]
        # (B, N, E) -> (N, B, E)
        x = x.transpose(0, 1)
        x, _ = self.lstm(x)
        # (N, B, E) -> (B, N, E)
        x = x.transpose(0, 1)
        return self.linear(x)
