"""
Train punctuation restoration model for Kazakh.
Reference: https://github.com/habib-wahid/punctuation-restoration

Usage:
  python -m src.train --model xlm-roberta-base --data-path ./kazakh_punct_data
"""

import argparse
import os
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from tqdm import tqdm
from collections import Counter

from src.config import MODELS, PUNCTUATION_DICT, MODEL_HF_IDS
from src.dataset import KazakhPunctDataset
from src.model import DeepPunctuation, DeepPunctuationCRF
from src.metrics import compute_metrics_from_batches


def parse_args():
    parser = argparse.ArgumentParser(description="Kazakh Punctuation Restoration")
    parser.add_argument("--model", type=str, default="xlm-roberta-base",
                        choices=list(MODELS.keys()),
                        help="Pretrained model name")
    parser.add_argument("--data-path", type=str, default="./kazakh_punct_data",
                        help="Path to train/val parquet files")
    parser.add_argument("--save-path", type=str, default="./out",
                        help="Where to save checkpoints and logs")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--sequence-length", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=5e-6)
    parser.add_argument("--unfreeze-lr", type=float, default=None,
                        help="LR for phase 2 (unfrozen BERT). Default: lr * 0.1")
    parser.add_argument("--freeze-bert", action="store_true",
                        help="Freeze BERT for all epochs (overrides two-phase)")
    parser.add_argument("--lstm-dim", type=int, default=-1,
                        help="-1 = use BERT hidden size")
    parser.add_argument("--dropout", type=float, default=0.1,
                        help="Dropout after BiLSTM")
    parser.add_argument("--use-crf", action="store_true",
                        help="Use CRF layer for sequence-aware predictions")
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cuda", action="store_true", default=True)
    return parser.parse_args()


def compute_class_weights(train_dataset):
    """Compute inverse-frequency class weights for imbalanced loss."""
    counts = Counter()
    for item in train_dataset.data:
        _, y, _, y_mask = item
        for i, m in enumerate(y_mask):
            if m == 1 and 0 <= y[i] < len(PUNCTUATION_DICT):
                counts[y[i]] += 1
    total = sum(counts.values())
    num_classes = len(PUNCTUATION_DICT)
    weights = torch.ones(num_classes)
    for c in range(num_classes):
        if counts[c] > 0:
            weights[c] = total / (num_classes * counts[c])
    return weights


def main():
    args = parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    device = torch.device("cuda" if (args.cuda and torch.cuda.is_available()) else "cpu")
    print(f"Using device: {device}")

    # Data
    data_path = Path(args.data_path)
    train_path = data_path / "train.parquet"
    val_path = data_path / "val.parquet"
    if not train_path.exists():
        raise FileNotFoundError(f"Train data not found at {train_path}. Run preprocessing first.")
    if not val_path.exists():
        raise FileNotFoundError(f"Val data not found at {val_path}. Run preprocessing first.")

    # Tokenizer and model config
    model_name = args.model
    hf_id = MODEL_HF_IDS.get(model_name, model_name)
    model_cls, tokenizer_cls, hidden_dim, token_style = MODELS[model_name]
    tokenizer = tokenizer_cls.from_pretrained(hf_id)
    num_classes = len(PUNCTUATION_DICT)

    # Datasets
    train_dataset = KazakhPunctDataset(
        str(train_path),
        tokenizer=tokenizer,
        sequence_len=args.sequence_length,
        token_style=token_style,
    )
    val_dataset = KazakhPunctDataset(
        str(val_path),
        tokenizer=tokenizer,
        sequence_len=args.sequence_length,
        token_style=token_style,
    )

    # Class weights for imbalanced loss (non-CRF)
    class_weights = compute_class_weights(train_dataset).to(device)
    print(f"Class weights: {class_weights.tolist()}")

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0
    )

    # Two-phase training: first half freeze BERT, second half unfreeze
    use_two_phase = not args.freeze_bert
    phase1_epochs = args.epochs // 2 if use_two_phase else args.epochs
    phase2_epochs = args.epochs - phase1_epochs if use_two_phase else 0

    # Model: start with frozen BERT (phase 1)
    model = DeepPunctuationCRF(model_name, freeze_bert=True, lstm_dim=args.lstm_dim, dropout=args.dropout) if args.use_crf else DeepPunctuation(model_name, freeze_bert=True, lstm_dim=args.lstm_dim, dropout=args.dropout)
    model.to(device)

    criterion = nn.CrossEntropyLoss(ignore_index=-100, weight=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # Training
    os.makedirs(args.save_path, exist_ok=True)
    log_path = os.path.join(args.save_path, "train_logs.txt")
    with open(log_path, "a") as f:
        f.write(str(args) + "\n")

    best_macro_f1 = 0.0
    # Save config for inference (whether CRF was used)
    torch.save({"use_crf": args.use_crf, "model_name": model_name}, os.path.join(args.save_path, "config.pt"))

    for epoch in range(args.epochs):
        # Two-phase: unfreeze BERT at start of phase 2
        if use_two_phase and epoch == phase1_epochs:
            bert = model.bert_lstm.bert_layer if args.use_crf else model.bert_layer
            for p in bert.parameters():
                p.requires_grad = True
            unfreeze_lr = args.unfreeze_lr if args.unfreeze_lr is not None else args.lr * 0.1
            for pg in optimizer.param_groups:
                pg["lr"] = unfreeze_lr
            print(f"[Phase 2] Unfreezing BERT, lr={unfreeze_lr}")

        model.train()
        train_loss = 0.0
        correct, total = 0, 0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            x = batch["input_ids"].to(device)
            att = batch["attention_mask"].to(device)
            y = batch["labels"].to(device)
            y_mask = batch["label_mask"].to(device)

            optimizer.zero_grad()
            if args.use_crf:
                loss = model.log_likelihood(x, att, y)
                pred = model.decode(x, att)
            else:
                logits = model(x, att)
                flat_logits = logits.view(-1, num_classes)
                flat_labels = y.view(-1).clone()
                flat_labels[~y_mask.view(-1).bool()] = -100
                loss = criterion(flat_logits, flat_labels)
                pred = torch.argmax(logits, dim=-1)

            loss.backward()
            if args.gradient_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clip)
            optimizer.step()

            train_loss += loss.item()
            pred = pred.view(-1)
            mask = y_mask.view(-1).bool()
            correct += (pred[mask] == y.view(-1)[mask]).sum().item()
            total += mask.sum().item()

        train_acc = correct / total if total > 0 else 0
        train_loss /= len(train_loader)
        log = f"Epoch {epoch+1}: Train loss={train_loss:.4f}, Train acc={train_acc:.4f}"
        print(log)
        with open(log_path, "a") as f:
            f.write(log + "\n")

        # Validation
        model.eval()
        all_preds, all_labels, all_masks = [], [], []
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                x = batch["input_ids"].to(device)
                att = batch["attention_mask"].to(device)
                y = batch["labels"].to(device)
                y_mask = batch["label_mask"].to(device)

                if args.use_crf:
                    pred = model.decode(x, att)
                    loss = model.log_likelihood(x, att, y)
                else:
                    logits = model(x, att)
                    pred = torch.argmax(logits, dim=-1)
                    flat_logits = logits.view(-1, num_classes)
                    flat_labels = y.view(-1).clone()
                    flat_labels[~y_mask.view(-1).bool()] = -100
                    loss = criterion(flat_logits, flat_labels)

                val_loss += loss.item()
                all_preds.append(pred.cpu().numpy())
                all_labels.append(y.cpu().numpy())
                all_masks.append(y_mask.cpu().numpy())

        preds = np.concatenate(all_preds)
        labels = np.concatenate(all_labels)
        masks = np.concatenate(all_masks)
        val_loss /= len(val_loader)

        metrics = compute_metrics_from_batches(preds, labels, masks, num_classes)

        log = (f"Epoch {epoch+1}: Val loss={val_loss:.4f}, "
               f"Acc={metrics['accuracy']:.4f}, "
               f"Macro F1 (excl O)={metrics['macro_f1']:.4f}, "
               f"F1={metrics['f1']:.4f}")
        print(log)
        with open(log_path, "a") as f:
            f.write(log + "\n")
            f.write(f"  Precision: {metrics['precision']}\n")
            f.write(f"  Recall: {metrics['recall']}\n")

        if metrics["macro_f1"] > best_macro_f1:
            best_macro_f1 = metrics["macro_f1"]
            ckpt_path = os.path.join(args.save_path, "best.pt")
            torch.save(model.state_dict(), ckpt_path)
            print(f"  Saved best model (macro F1 excl O: {best_macro_f1:.4f})")

    print(f"\nBest Macro F1 (excl O): {best_macro_f1:.4f}")
    print(f"Logs: {log_path}")


if __name__ == "__main__":
    main()
