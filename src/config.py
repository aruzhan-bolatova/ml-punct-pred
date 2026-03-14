"""
Config for Kazakh punctuation restoration pipeline.
Reference: https://github.com/habib-wahid/punctuation-restoration
"""

from transformers import (
    BertModel,
    BertTokenizer,
    RobertaModel,
    RobertaTokenizer,
    XLMRobertaModel,
    XLMRobertaTokenizer,
)

# Special token indices for different model families
TOKEN_IDX = {
    "bert": {"START_SEQ": 101, "PAD": 0, "END_SEQ": 102, "UNK": 100},
    "xlm": {"START_SEQ": 0, "PAD": 2, "END_SEQ": 1, "UNK": 3},
    "roberta": {"START_SEQ": 0, "PAD": 1, "END_SEQ": 2, "UNK": 3},
}

# Label mapping: O=0, COMMA=1, PERIOD=2, QUESTION=3
PUNCTUATION_DICT = {"O": 0, "COMMA": 1, "PERIOD": 2, "QUESTION": 3}
ID2LABEL = {v: k for k, v in PUNCTUATION_DICT.items()}

# Model registry: (model_class, tokenizer_class, hidden_dim, token_style)
# token_style: bert | xlm | roberta (for TOKEN_IDX)
MODELS = {
    # Default + multilingual
    "xlm-roberta-base": (XLMRobertaModel, XLMRobertaTokenizer, 768, "roberta"),
    "xlm-roberta-large": (XLMRobertaModel, XLMRobertaTokenizer, 1024, "roberta"),
    "bert-base-multilingual-cased": (BertModel, BertTokenizer, 768, "bert"),
    # Kazakh-specific
    "kaz-roberta": (
        RobertaModel,
        RobertaTokenizer,
        768,
        "roberta",
    ),  # kz-transformers/kaz-roberta-conversational
    "kazbert": (
        RobertaModel,
        RobertaTokenizer,
        1024,
        "roberta",
    ),  # nur-dev/roberta-kaz-large
    "bert-base-kazakh": (
        BertModel,
        BertTokenizer,
        768,
        "bert",
    ),  # placeholder; use multilingual if not available
}

# Full HuggingFace model IDs for Kazakh models
MODEL_HF_IDS = {
    "xlm-roberta-base": "xlm-roberta-base",
    "xlm-roberta-large": "xlm-roberta-large",
    "bert-base-multilingual-cased": "bert-base-multilingual-cased",
    "kaz-roberta": "kz-transformers/kaz-roberta-conversational",
    "kazbert": "nur-dev/roberta-kaz-large",
    "bert-base-kazakh": "bert-base-multilingual-cased",  # fallback; replace if Kazakh BERT exists
}
