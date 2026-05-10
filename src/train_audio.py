"""
train_audio.py
--------------
Standalone training script for the audio branch (AudioFERModel / Wav2Vec2).

Usage:
    python src/train_audio.py --data_root audio_data/ --dataset ravdess --epochs 20

Supports RAVDESS, CREMA-D, or a folder-organised dataset (0-6 sub-folders).
See audio_model.SpeechEmotionDataset for expected layouts.
"""

import os
import sys
import time
import argparse
import json

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

sys.path.insert(0, os.path.dirname(__file__))
from audio_model import AudioFERModel, build_audio_dataloaders, EMOTION_NAMES
from models import count_trainable_params


def parse_args():
    p = argparse.ArgumentParser(description="Train audio emotion model (Wav2Vec2)")
    p.add_argument("--data_root",  required=True,
                   help="Root folder of audio dataset (RAVDESS / CREMA-D / folder)")
    p.add_argument("--dataset",    default=None,
                   choices=["ravdess", "cremad", "folder", None],
                   help="Dataset type (auto-detected if omitted)")
    p.add_argument("--epochs",     type=int,   default=20)
    p.add_argument("--batch_size", type=int,   default=16)
    p.add_argument("--lr",         type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--label_smoothing", type=float, default=0.1)
    p.add_argument("--grad_clip",  type=float, default=1.0)
    p.add_argument("--val_split",  type=float, default=0.1)
    p.add_argument("--num_workers", type=int,  default=0)
    p.add_argument("--unfreeze",   action="store_true",
                   help="Unfreeze Wav2Vec2 backbone (needs GPU + time)")
    p.add_argument("--output_dir", default="checkpoints",
                   help="Where to save checkpoints")
    return p.parse_args()


def accuracy(logits, labels):
    return (logits.argmax(dim=1) == labels).float().mean().item()


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = total_acc = n = 0
    all_preds, all_labels = [], []
    for audio, labels in loader:
        audio, labels = audio.to(device), labels.to(device)
        logits = model(audio)
        total_loss += criterion(logits, labels).item()
        total_acc  += accuracy(logits, labels)
        n += 1
        all_preds.extend(logits.argmax(1).cpu().tolist())
        all_labels.extend(labels.cpu().tolist())
    return total_loss / max(n, 1), total_acc / max(n, 1), all_preds, all_labels


def macro_f1(preds, labels, num_classes=7):
    tp = [0] * num_classes
    fp = [0] * num_classes
    fn = [0] * num_classes
    for p, l in zip(preds, labels):
        if p == l:
            tp[l] += 1
        else:
            fp[p] += 1
            fn[l] += 1
    f1s = []
    for c in range(num_classes):
        denom = 2 * tp[c] + fp[c] + fn[c]
        f1s.append(2 * tp[c] / denom if denom > 0 else 0.0)
    return sum(f1s) / num_classes


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*60}")
    print(f"  Device  : {device}")
    print(f"  Dataset : {args.data_root}")
    print(f"  Epochs  : {args.epochs}")
    print(f"{'='*60}\n")

    train_loader, val_loader = build_audio_dataloaders(
        root_dir    = args.data_root,
        dataset     = args.dataset,
        batch_size  = args.batch_size,
        val_split   = args.val_split,
        num_workers = args.num_workers,
    )

    model = AudioFERModel(freeze_backbone=not args.unfreeze).to(device)
    print(f"Trainable parameters : {count_trainable_params(model):,}\n")

    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                      lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs * len(train_loader),
                                   eta_min=1e-6)

    best_val_acc = 0.0
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    for epoch in range(1, args.epochs + 1):
        model.train()
        ep_loss = ep_acc = n = 0
        t0 = time.time()

        for audio, labels in train_loader:
            audio, labels = audio.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = model(audio)
            loss   = criterion(logits, labels)
            loss.backward()
            if args.grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            scheduler.step()
            ep_loss += loss.item()
            ep_acc  += accuracy(logits, labels)
            n += 1

        train_loss = ep_loss / max(n, 1)
        train_acc  = ep_acc  / max(n, 1)
        val_loss, val_acc, _, _ = evaluate(model, val_loader, criterion, device)

        print(f"Epoch {epoch:02d}/{args.epochs}  "
              f"train_loss={train_loss:.4f}  train_acc={train_acc:.4f}  "
              f"val_loss={val_loss:.4f}  val_acc={val_acc:.4f}  "
              f"({time.time()-t0:.0f}s)")

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            ckpt = os.path.join(args.output_dir, "best_audio.pt")
            torch.save(model.state_dict(), ckpt)
            print(f"  [BEST] val_acc={best_val_acc:.4f} -> {ckpt}")

    # ── Final val F1 ──────────────────────────────────────────────────────────
    ckpt = os.path.join(args.output_dir, "best_audio.pt")
    if os.path.exists(ckpt):
        model.load_state_dict(torch.load(ckpt, map_location=device))
    _, final_acc, preds, labels = evaluate(model, val_loader, criterion, device)
    f1 = macro_f1(preds, labels)

    print(f"\n  Val Accuracy : {final_acc:.4f}   Macro F1 : {f1:.4f}")

    results = {"val_acc": round(final_acc, 4), "macro_f1": round(f1, 4),
               "best_val_acc": round(best_val_acc, 4), "history": history}
    rp = os.path.join(args.output_dir, "results_audio.json")
    with open(rp, "w") as fp:
        json.dump(results, fp, indent=2)
    print(f"  Results -> {rp}\n")


if __name__ == "__main__":
    main()
