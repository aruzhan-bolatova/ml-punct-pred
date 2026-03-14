"""
Dataset for Kazakh punctuation restoration.
Converts (input_text, labels) format to word-level sequences for the model.
"""

import os
import torch
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Union, Optional

try:
    from .config import TOKEN_IDX, PUNCTUATION_DICT
except ImportError:
    from config import TOKEN_IDX, PUNCTUATION_DICT


def parse_data_from_df(
    df: pd.DataFrame,
    tokenizer,
    sequence_len: int,
    token_style: str,
) -> List[List]:
    """
    Parse dataframe with input_text + labels into model-ready sequences.
    Format: each row has "word word word" and "O COMMA PERIOD" (space-separated).
    Returns list of [x, y, attn_mask, y_mask] where y_mask=1 for word-end positions.
    """
    data_items = []
    text_col = "input_text" if "input_text" in df.columns else df.columns[0]
    label_col = "labels" if "labels" in df.columns else df.columns[1]

    for _, row in df.iterrows():
        words = str(row[text_col]).strip().split()
        labels_str = str(row[label_col]).strip().split()
        if len(words) != len(labels_str) or len(words) == 0:
            continue

        # Build sequence: [START] + word_tokens + [END], labels at word-end positions
        x = [TOKEN_IDX[token_style]["START_SEQ"]]
        y = [0]
        y_mask = [1]

        idx = 0
        while idx < len(words) and len(x) < sequence_len - 1:
            word = words[idx]
            label_str = labels_str[idx].upper()
            if label_str not in PUNCTUATION_DICT:
                label_str = "O"
            label_id = PUNCTUATION_DICT[label_str]

            tokens = tokenizer.tokenize(word)
            if len(tokens) + len(x) >= sequence_len:
                break

            for i in range(len(tokens) - 1):
                x.append(tokenizer.convert_tokens_to_ids(tokens[i]))
                y.append(0)
                y_mask.append(0)
            if len(tokens) > 0:
                x.append(tokenizer.convert_tokens_to_ids(tokens[-1]))
            else:
                x.append(TOKEN_IDX[token_style]["UNK"])
            y.append(label_id)
            y_mask.append(1)
            idx += 1

        x.append(TOKEN_IDX[token_style]["END_SEQ"])
        y.append(0)
        y_mask.append(1)

        # Pad to sequence_len
        if len(x) < sequence_len:
            x = x + [TOKEN_IDX[token_style]["PAD"]] * (sequence_len - len(x))
            y = y + [0] * (sequence_len - len(y))
            y_mask = y_mask + [0] * (sequence_len - len(y_mask))
        attn_mask = [1 if t != TOKEN_IDX[token_style]["PAD"] else 0 for t in x]
        data_items.append([x, y, attn_mask, y_mask])

    return data_items


def parse_data_from_files(
    file_paths: Union[str, List[str]],
    tokenizer,
    sequence_len: int,
    token_style: str,
) -> List[List]:
    """
    Parse from text files: one line per sentence, format "word\tlabel" per line
    or "word word" / "label label" in alternating blocks.
    For standard format: word\tO, word\tCOMMA, etc. per line.
    """
    if isinstance(file_paths, str):
        file_paths = [file_paths]

    data_items = []
    for path in file_paths:
        with open(path, "r", encoding="utf-8") as f:
            lines = [line.strip() for line in f if line.strip()]

        idx = 0
        while idx < len(lines):
            x = [TOKEN_IDX[token_style]["START_SEQ"]]
            y = [0]
            y_mask = [1]

            while len(x) < sequence_len - 1 and idx < len(lines):
                parts = lines[idx].split("\t")
                if len(parts) != 2:
                    idx += 1
                    continue
                word, punc = parts[0].strip(), parts[1].strip().upper()
                if punc not in PUNCTUATION_DICT:
                    punc = "O"

                tokens = tokenizer.tokenize(word)
                if len(tokens) + len(x) >= sequence_len:
                    break
                for i in range(len(tokens) - 1):
                    x.append(tokenizer.convert_tokens_to_ids(tokens[i]))
                    y.append(0)
                    y_mask.append(0)
                if len(tokens) > 0:
                    x.append(tokenizer.convert_tokens_to_ids(tokens[-1]))
                else:
                    x.append(TOKEN_IDX[token_style]["UNK"])
                y.append(PUNCTUATION_DICT[punc])
                y_mask.append(1)
                idx += 1

            x.append(TOKEN_IDX[token_style]["END_SEQ"])
            y.append(0)
            y_mask.append(1)
            if len(x) < sequence_len:
                x = x + [TOKEN_IDX[token_style]["PAD"]] * (sequence_len - len(x))
                y = y + [0] * (sequence_len - len(y))
                y_mask = y_mask + [0] * (sequence_len - len(y_mask))
            attn_mask = [1 if t != TOKEN_IDX[token_style]["PAD"] else 0 for t in x]
            data_items.append([x, y, attn_mask, y_mask])

    return data_items


class KazakhPunctDataset(torch.utils.data.Dataset):
    """Dataset for Kazakh punctuation prediction."""

    def __init__(
        self,
        data_path: Optional[Union[str, Path, pd.DataFrame]] = None,
        tokenizer=None,
        sequence_len: int = 128,
        token_style: str = "roberta",
        is_train: bool = False,
    ):
        """
        Args:
            data_path: path to parquet/csv, or DataFrame directly
            tokenizer: HuggingFace tokenizer
            sequence_len: max sequence length
            token_style: 'bert' | 'roberta' | 'xlm'
            is_train: unused, for API compatibility
        """
        if isinstance(data_path, pd.DataFrame):
            df = data_path
        elif isinstance(data_path, (str, Path)):
            path = Path(data_path)
            if path.suffix in (".parquet", ".pq"):
                df = pd.read_parquet(path)
            else:
                df = pd.read_csv(path)
        else:
            raise ValueError("data_path must be DataFrame, str, or Path")

        self.data = parse_data_from_df(df, tokenizer, sequence_len, token_style)
        self.sequence_len = sequence_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x, y, attn, y_mask = self.data[idx]
        return {
            "input_ids": torch.tensor(x, dtype=torch.long),
            "labels": torch.tensor(y, dtype=torch.long),
            "attention_mask": torch.tensor(attn, dtype=torch.long),
            "label_mask": torch.tensor(y_mask, dtype=torch.long),
        }
