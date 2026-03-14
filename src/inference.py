"""
Inference for Kazakh punctuation restoration.
Predicts labels for test.csv (id, input_text) using a trained best.pt model.
Output: submission CSV with id, labels (space-separated O/COMMA/PERIOD/QUESTION).
"""

import torch
import pandas as pd
from pathlib import Path
from typing import List
from tqdm import tqdm

from src.config import MODELS, MODEL_HF_IDS, TOKEN_IDX, ID2LABEL


def build_sequences_for_inference(words: List[str], tokenizer, sequence_len: int, token_style: str):
    """
    Build (input_ids, attention_mask, word_end_positions) for inference.
    Handles long sentences by chunking. Returns list of (x, attn, positions)
    where positions are indices into the sequence where we predict (word-end).
    """
    sequences = []
    idx = 0
    while idx < len(words):
        x = [TOKEN_IDX[token_style]["START_SEQ"]]
        word_end_positions = []

        while idx < len(words) and len(x) < sequence_len - 1:
            word = words[idx]
            tokens = tokenizer.tokenize(word)
            if len(tokens) + len(x) >= sequence_len:
                break
            for i in range(len(tokens) - 1):
                x.append(tokenizer.convert_tokens_to_ids(tokens[i]))
            if len(tokens) > 0:
                x.append(tokenizer.convert_tokens_to_ids(tokens[-1]))
                word_end_positions.append(len(x) - 1)
            else:
                x.append(TOKEN_IDX[token_style]["UNK"])
                word_end_positions.append(len(x) - 1)
            idx += 1

        x.append(TOKEN_IDX[token_style]["END_SEQ"])
        if len(x) < sequence_len:
            x = x + [TOKEN_IDX[token_style]["PAD"]] * (sequence_len - len(x))
        attn = [1 if t != TOKEN_IDX[token_style]["PAD"] else 0 for t in x]
        sequences.append((x, attn, word_end_positions))
    return sequences


def predict_sentence(
    input_text: str,
    model,
    tokenizer,
    device,
    sequence_len: int = 256,
    token_style: str = "roberta",
    use_crf: bool = False,
) -> str:
    """
    Predict punctuation labels for a sentence. Returns space-separated labels.
    """
    words = input_text.strip().split()
    if not words:
        return ""

    seqs = build_sequences_for_inference(words, tokenizer, sequence_len, token_style)
    all_labels = []
    for x, attn, positions in seqs:
        x_t = torch.tensor([x], dtype=torch.long).to(device)
        attn_t = torch.tensor([attn], dtype=torch.long).to(device)
        with torch.no_grad():
            if use_crf:
                decoded = model.decode(x_t, attn_t)  # (B, N) tensor
                for p in positions:
                    lid = int(decoded[0, p].item())
                    all_labels.append(ID2LABEL.get(lid, "O"))
            else:
                logits = model(x_t, attn_t)
                preds = torch.argmax(logits, dim=-1).squeeze(0).cpu().numpy()
                for p in positions:
                    lid = int(preds[p])
                    all_labels.append(ID2LABEL.get(lid, "O"))
    return " ".join(all_labels)


def run_inference(
    model_name: str = "xlm-roberta-base",
    weight_path: str = "./out/best.pt",
    test_csv: str = "kaz-punct-hackathon/test.csv",
    sequence_len: int = 256,
    use_crf: bool = None,
) -> pd.DataFrame:
    """
    Run inference on test.csv. Returns DataFrame with id, labels.
    use_crf: If None, auto-detect from config.pt; else use this value.
    """
    return run_test(
        test_path=test_csv,
        weight_path=weight_path,
        out_path=None,
        model_name=model_name,
        sequence_len=sequence_len,
        use_crf=use_crf,
    )


def run_test(
    test_path: str,
    weight_path: str,
    out_path: str,
    model_name: str = "xlm-roberta-base",
    sequence_len: int = 256,
    batch_mode: bool = False,
    use_crf: bool = None,
):
    """
    Run inference on test.csv and save submission.
    use_crf: If None, auto-detect from config.pt in save dir.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_df = pd.read_csv(test_path)

    # Auto-detect settings from config.pt (use_crf, model_name, sequence_length)
    save_dir = Path(weight_path).parent
    config_path = save_dir / "config.pt"
    if config_path.exists():
        config = torch.load(config_path, map_location="cpu", weights_only=False)
        if use_crf is None:
            use_crf = config.get("use_crf", False)
        if "model_name" in config:
            model_name = config["model_name"]
        if "sequence_length" in config:
            sequence_len = config["sequence_length"]
    if use_crf is None:
        use_crf = False

    # Load model
    _, tokenizer_cls, _, token_style = MODELS[model_name]
    hf_id = MODEL_HF_IDS.get(model_name, model_name)
    tokenizer = tokenizer_cls.from_pretrained(hf_id)
    try:
        from src.model import DeepPunctuation, DeepPunctuationCRF
    except ImportError:
        from model import DeepPunctuation, DeepPunctuationCRF
    model = DeepPunctuationCRF(model_name) if use_crf else DeepPunctuation(model_name)
    model.load_state_dict(torch.load(weight_path, map_location=device))
    model.to(device)
    model.eval()

    predictions = []
    for _, row in tqdm(test_df.iterrows(), total=len(test_df), desc="Predicting"):
        rid = row["id"]
        inp = str(row["input_text"])
        labels_str = predict_sentence(
            inp, model, tokenizer, device,
            sequence_len=sequence_len,
            token_style=token_style,
            use_crf=use_crf,
        )
        predictions.append({"id": rid, "labels": labels_str})

    out_df = pd.DataFrame(predictions)
    if out_path:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        out_df.to_csv(out_path, index=False)
        print(f"Saved {len(out_df)} predictions to {out_path}")
    return out_df
