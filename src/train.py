"""
train.py
--------
Full training + evaluation loop for the FER models.

Usage (local CPU smoke-test):
    python train.py --model dino --epochs 1 --batch_size 8 --max_train_samples 64

Usage (Colab / GPU full run):
    python train.py --model convnext --epochs 20 --batch_size 32 --unfreeze
"""

import os
import sys
import time
import argparse
import json
import math

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

# Allow running from repo root
sys.path.insert(0, os.path.dirname(__file__))

from dataset import build_dataloaders, LABEL_MAP
from models import ConvNeXtFERModel, DINOFERModel, count_trainable_params

# ─── configuration ────────────────────────────────────────────────────────────

EMOTION_NAMES = [LABEL_MAP[i + 1] for i in range(7)]   # 0-indexed list


def parse_args():
    p = argparse.ArgumentParser(description="Train FER model")

    # Model
    p.add_argument("--model", choices=["dino", "convnext"], default="dino",
                   help="Which backbone to use")
    p.add_argument("--dino_variant", choices=["vits16", "vitb16"], default="vits16")
    p.add_argument("--unfreeze", action="store_true",
                   help="Fully fine-tune the backbone (needs GPU)")
    p.add_argument("--unfreeze_last_n", type=int, default=2,
                   help="When partially frozen, unfreeze last N transformer blocks")

    # Data
    p.add_argument("--data_root", default=".",
                   help="Root folder containing DATASET/ and train_labels.csv")
    p.add_argument("--image_size", type=int, default=224)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--val_split", type=float, default=0.1)
    p.add_argument("--max_train_samples", type=int, default=None,
                   help="Limit training samples for CPU smoke-test (e.g. 64)")

    # Training
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--label_smoothing", type=float, default=0.1)
    p.add_argument("--grad_clip", type=float, default=1.0)

    # Output
    p.add_argument("--output_dir", default="checkpoints",
                   help="Where to save model checkpoints and results")

    return p.parse_args()


# ─── metric helpers ───────────────────────────────────────────────────────────

def accuracy(preds, labels):
    return (preds.argmax(dim=1) == labels).float().mean().item()


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, total_acc, n_batches = 0.0, 0.0, 0
    all_preds, all_labels = [], []

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        logits = model(images)
        loss = criterion(logits, labels)
        total_loss += loss.item()
        total_acc  += accuracy(logits, labels)
        n_batches  += 1
        all_preds.extend(logits.argmax(dim=1).cpu().tolist())
        all_labels.extend(labels.cpu().tolist())

    avg_loss = total_loss / max(n_batches, 1)
    avg_acc  = total_acc  / max(n_batches, 1)
    return avg_loss, avg_acc, all_preds, all_labels


# ─── class-level F1 (macro) without sklearn dependency ───────────────────────

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


# ─── per-class accuracy breakdown ────────────────────────────────────────────

def class_accuracy(preds, labels, num_classes=7):
    correct = [0] * num_classes
    total   = [0] * num_classes
    for p, l in zip(preds, labels):
        total[l] += 1
        if p == l:
            correct[l] += 1
    return [correct[c] / max(total[c], 1) for c in range(num_classes)]


# ─── main training function ───────────────────────────────────────────────────

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*60}")
    print(f"  Device  : {device}")
    print(f"  Model   : {args.model}")
    print(f"  Epochs  : {args.epochs}")
    print(f"  Batch   : {args.batch_size}")
    print(f"{'='*60}\n")

    # ── Build data loaders ────────────────────────────────────────────────────
    csv_path    = os.path.join(args.data_root, "train_labels.csv")
    train_img   = os.path.join(args.data_root, "DATASET", "train")
    test_dir    = os.path.join(args.data_root, "DATASET", "test")

    train_loader, val_loader, test_loader = build_dataloaders(
        csv_path      = csv_path,
        train_img_dir = train_img,
        test_dir      = test_dir,
        batch_size    = args.batch_size,
        num_workers   = args.num_workers,
        image_size    = args.image_size,
        val_split     = args.val_split,
    )

    # Limit samples for local CPU smoke-testing
    if args.max_train_samples:
        from itertools import islice
        class _Limited:
            def __init__(self, loader, n):
                self._loader = loader
                self._n = n
            def __iter__(self):
                return islice(iter(self._loader), math.ceil(self._n / self._loader.batch_size))
            def __len__(self):
                return math.ceil(self._n / self._loader.batch_size)
        train_loader = _Limited(train_loader, args.max_train_samples)
        print(f"[SMOKE-TEST] Limiting training to {args.max_train_samples} samples")

    # ── Build model ───────────────────────────────────────────────────────────
    if args.model == "dino":
        model = DINOFERModel(
            model_variant=args.dino_variant,
            freeze=not args.unfreeze,
            unfreeze_last_n_blocks=args.unfreeze_last_n,
        )
    else:  # convnext
        model = ConvNeXtFERModel(freeze_backbone=not args.unfreeze)

    model = model.to(device)
    trainable = count_trainable_params(model)
    print(f"Trainable parameters : {trainable:,}\n")

    # ── Optimiser + Scheduler + Loss ─────────────────────────────────────────
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr, weight_decay=args.weight_decay
    )
    total_steps = args.epochs * len(train_loader)
    scheduler = CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=1e-6)

    # ── Training loop ─────────────────────────────────────────────────────────
    best_val_acc = 0.0
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    for epoch in range(1, args.epochs + 1):
        model.train()
        ep_loss, ep_acc, n = 0.0, 0.0, 0
        t0 = time.time()

        for step, (images, labels) in enumerate(train_loader, 1):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            logits = model(images)
            loss   = criterion(logits, labels)
            loss.backward()

            if args.grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

            optimizer.step()
            scheduler.step()

            ep_loss += loss.item()
            ep_acc  += accuracy(logits, labels)
            n += 1

            if step % max(1, len(train_loader) // 4) == 0:
                print(f"  [Ep {epoch}/{args.epochs} | Step {step}/{len(train_loader)}] "
                      f"loss={ep_loss/n:.4f}  acc={ep_acc/n:.3f}  "
                      f"lr={scheduler.get_last_lr()[0]:.2e}")

        train_loss = ep_loss / max(n, 1)
        train_acc  = ep_acc  / max(n, 1)

        # Validation
        val_loss, val_acc, _, _ = evaluate(model, val_loader, criterion, device)

        elapsed = time.time() - t0
        print(f"\nEpoch {epoch:02d}/{args.epochs:02d}  "
              f"train_loss={train_loss:.4f}  train_acc={train_acc:.4f}  "
              f"val_loss={val_loss:.4f}  val_acc={val_acc:.4f}  "
              f"({elapsed:.0f}s)\n")

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        # Save best checkpoint
        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            ckpt_path = os.path.join(args.output_dir, f"best_{args.model}.pt")
            torch.save(model.state_dict(), ckpt_path)
            print(f"  [BEST] New best val_acc={best_val_acc:.4f} -> saved to {ckpt_path}\n")

    # ── Final test evaluation ─────────────────────────────────────────────────
    print("\n" + "="*60)
    print("FINAL TEST EVALUATION")
    print("="*60)

    best_ckpt = os.path.join(args.output_dir, f"best_{args.model}.pt")
    if os.path.exists(best_ckpt):
        model.load_state_dict(torch.load(best_ckpt, map_location=device))
        print(f"Loaded best checkpoint: {best_ckpt}")

    _, test_acc, test_preds, test_labels = evaluate(
        model, test_loader, criterion, device
    )
    f1 = macro_f1(test_preds, test_labels)
    per_class = class_accuracy(test_preds, test_labels)

    print(f"\n  Test Accuracy : {test_acc:.4f}  ({test_acc*100:.2f}%)")
    print(f"  Macro F1      : {f1:.4f}")
    print(f"\n  Per-class accuracy:")
    for i, (name, acc) in enumerate(zip(EMOTION_NAMES, per_class)):
        bar = "█" * int(acc * 20)
        print(f"    {name:10s} [{bar:<20s}] {acc*100:.1f}%")

    # ── Save results ──────────────────────────────────────────────────────────
    results = {
        "model"       : args.model,
        "test_acc"    : round(test_acc, 4),
        "macro_f1"    : round(f1, 4),
        "best_val_acc": round(best_val_acc, 4),
        "per_class_acc": {n: round(a, 4) for n, a in zip(EMOTION_NAMES, per_class)},
        "history"     : history,
    }
    results_path = os.path.join(args.output_dir, f"results_{args.model}.json")
    with open(results_path, "w", encoding="utf-8") as fp:
        json.dump(results, fp, indent=2)
    print(f"\n  Results saved -> {results_path}")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
