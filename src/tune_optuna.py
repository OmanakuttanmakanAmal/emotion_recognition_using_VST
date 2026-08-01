"""
tune_optuna.py
--------------
Hyperparameter search for the face (DINO) model using Optuna.

What Optuna does
-----------------
Instead of hand-picking --lr / --weight_decay / --label_smoothing / --unfreeze_last_n
(the defaults baked into train.py), Optuna runs many short trials with different
combinations of those values, and uses a Bayesian-style search (TPE sampler) to
zero in on the combination that maximises validation accuracy -- much cheaper
than a manual grid search, and better than guessing.

Each trial here trains for only a handful of epochs (--epochs_per_trial) on the
same data the real run will use, so the search itself is cheap. A pruner kills
clearly-bad trials early (e.g. a bad learning rate that isn't improving) instead
of wasting the full epoch budget on it.

Usage:
    python src/tune_optuna.py --n_trials 25 --epochs_per_trial 5

Then take study best_params and pass them into the real full run:
    python src/train.py --model dino --epochs 20 --lr <best_lr> \\
        --weight_decay <best_weight_decay> --label_smoothing <best_label_smoothing> \\
        --unfreeze_last_n <best_unfreeze_last_n>

Requires: pip install optuna
"""

import os
import sys
import json
import argparse
from itertools import islice

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

import optuna
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner

sys.path.insert(0, os.path.dirname(__file__))
from dataset import build_dataloaders
from models import DINOFERModel


class _Limited:
    """Caps a loader to the first n samples per epoch (faster search trials)."""

    def __init__(self, loader, n):
        self._loader = loader
        self._n = n

    def __iter__(self):
        return islice(iter(self._loader), -(-self._n // self._loader.batch_size))

    def __len__(self):
        return -(-self._n // self._loader.batch_size)


def accuracy(logits, labels):
    return (logits.argmax(dim=1) == labels).float().mean().item()


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = total_acc = n = 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        logits = model(images)
        total_loss += criterion(logits, labels).item()
        total_acc += accuracy(logits, labels)
        n += 1
    return total_loss / max(n, 1), total_acc / max(n, 1)


def parse_args():
    p = argparse.ArgumentParser(description="Optuna hyperparameter search for the DINO face model")
    p.add_argument("--data_root", default=".")
    p.add_argument("--image_size", type=int, default=224)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--val_split", type=float, default=0.1)
    p.add_argument("--n_trials", type=int, default=25)
    p.add_argument("--epochs_per_trial", type=int, default=5)
    p.add_argument("--max_train_samples", type=int, default=None,
                    help="Subsample training data per trial for a faster search")
    p.add_argument("--output_dir", default="checkpoints")
    return p.parse_args()


def build_objective(args, device):
    csv_path = os.path.join(args.data_root, "train_labels.csv")
    train_img = os.path.join(args.data_root, "DATASET", "train")
    test_dir = os.path.join(args.data_root, "DATASET", "test")

    train_loader, val_loader, _ = build_dataloaders(
        csv_path=csv_path,
        train_img_dir=train_img,
        test_dir=test_dir,
        batch_size=args.batch_size,
        image_size=args.image_size,
        val_split=args.val_split,
    )
    if args.max_train_samples:
        train_loader = _Limited(train_loader, args.max_train_samples)

    def objective(trial):
        lr = trial.suggest_float("lr", 1e-5, 1e-3, log=True)
        weight_decay = trial.suggest_float("weight_decay", 1e-4, 1e-1, log=True)
        label_smoothing = trial.suggest_float("label_smoothing", 0.0, 0.2)
        dropout = trial.suggest_float("dropout", 0.1, 0.5)
        unfreeze_last_n = trial.suggest_categorical("unfreeze_last_n", [0, 2, 4])

        model = DINOFERModel(
            freeze=True,
            unfreeze_last_n_blocks=unfreeze_last_n,
            dropout=dropout,
        ).to(device)

        criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        optimizer = AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=lr, weight_decay=weight_decay,
        )
        total_steps = args.epochs_per_trial * len(train_loader)
        scheduler = CosineAnnealingLR(optimizer, T_max=max(total_steps, 1), eta_min=1e-6)

        best_val_acc = 0.0
        for epoch in range(1, args.epochs_per_trial + 1):
            model.train()
            for images, labels in train_loader:
                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad()
                logits = model(images)
                loss = criterion(logits, labels)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()

            _, val_acc = evaluate(model, val_loader, criterion, device)
            best_val_acc = max(best_val_acc, val_acc)

            trial.report(val_acc, epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()

        return best_val_acc

    return objective


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    objective = build_objective(args, device)
    study = optuna.create_study(
        direction="maximize",
        sampler=TPESampler(seed=42),
        pruner=MedianPruner(n_warmup_steps=2),
    )
    study.optimize(objective, n_trials=args.n_trials)

    print("\n" + "=" * 60)
    print(f"Best val_acc : {study.best_value:.4f}")
    print(f"Best params  : {study.best_params}")
    print("=" * 60)

    out_path = os.path.join(args.output_dir, "optuna_best_params.json")
    with open(out_path, "w", encoding="utf-8") as fp:
        json.dump({"best_value": study.best_value, "best_params": study.best_params}, fp, indent=2)
    print(f"\nSaved -> {out_path}")


if __name__ == "__main__":
    main()
