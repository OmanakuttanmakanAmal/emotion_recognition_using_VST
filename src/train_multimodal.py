"""
train_multimodal.py
-------------------
Train the late-fusion MultimodalFusionModel using pre-trained face and audio
branch checkpoints.

Strategy
--------
1. Load a trained DINOFERModel (face) and AudioFERModel (audio) — both frozen
   as feature extractors.
2. For every training sample, extract face_feat (384-d) and audio_feat (768-d).
3. Train only the MultimodalFusionModel MLP on the concatenated embeddings.

Paired data requirement
-----------------------
The fusion model needs aligned face images and audio clips for the SAME
utterance.  Two dataset layouts are supported:

  Layout A — "parallel CSVs"  (default):
      fusion_labels.csv with columns: image, audio, label
      Images live in DATASET/train/<class>/
      Audio files live in audio_data/<class>/

  Layout B — "synthetic pairing"  (--synthetic):
      Randomly pairs face and audio samples from the SAME class within a batch.
      Useful when you don't have perfectly aligned pairs; gives a lower bound
      on fusion accuracy but lets you test the pipeline end-to-end immediately.

Usage:
    # With real paired data
    python src/train_multimodal.py \\
        --face_ckpt  checkpoints/best_dino.pt \\
        --audio_ckpt checkpoints/best_audio.pt \\
        --csv        fusion_labels.csv \\
        --img_root   DATASET/train \\
        --audio_root audio_data \\
        --epochs 20

    # With synthetic pairing (no paired dataset needed)
    python src/train_multimodal.py \\
        --face_ckpt  checkpoints/best_dino.pt \\
        --audio_ckpt checkpoints/best_audio.pt \\
        --synthetic \\
        --n_synthetic 2000 \\
        --epochs 20
"""

import os
import sys
import csv
import time
import argparse
import json
import random

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset
from PIL import Image

sys.path.insert(0, os.path.dirname(__file__))
from dataset    import get_val_transforms, LABEL_MAP
from models     import DINOFERModel, MultimodalFusionModel, count_trainable_params
from audio_model import AudioFERModel, _load_waveform

EMOTION_NAMES = [LABEL_MAP[i + 1] for i in range(7)]
NUM_CLASSES   = 7


# ─── Paired dataset ───────────────────────────────────────────────────────────
class PairedFaceAudioDataset(Dataset):
    """Reads aligned (image, audio, label) triples from a CSV."""

    def __init__(self, csv_path: str, img_root: str, audio_root: str,
                 image_size: int = 224, sample_rate: int = 16000):
        self.img_root    = img_root
        self.audio_root  = audio_root
        self.transform   = get_val_transforms(image_size)
        self.sample_rate = sample_rate
        self.samples: list[tuple[str, str, int]] = []

        with open(csv_path, newline="") as f:
            for row in csv.DictReader(f):
                img   = row["image"].strip()
                audio = row["audio"].strip()
                lbl   = int(row["label"].strip()) - 1   # 1-indexed → 0-indexed
                class_dir = str(int(row["label"].strip()))
                img_path   = os.path.join(img_root,   class_dir, img)
                audio_path = os.path.join(audio_root, class_dir, audio)
                self.samples.append((img_path, audio_path, lbl))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, audio_path, label = self.samples[idx]
        image    = Image.open(img_path).convert("RGB")
        image    = self.transform(image)
        waveform = _load_waveform(audio_path, self.sample_rate)
        return image, waveform, label


# ─── Synthetic pairing dataset ────────────────────────────────────────────────
class SyntheticPairedDataset(Dataset):
    """
    Generates fake paired embeddings from the feature extractors.
    Used to test/train the fusion head without aligned A/V data.
    Embeddings are pre-computed once at construction time.
    """

    def __init__(self, face_model: nn.Module, audio_model: nn.Module,
                 face_data, audio_data, n_samples: int, device: torch.device):
        face_model.eval()
        audio_model.eval()

        # Bucket face and audio features by class
        face_by_class:  dict[int, list] = {c: [] for c in range(NUM_CLASSES)}
        audio_by_class: dict[int, list] = {c: [] for c in range(NUM_CLASSES)}

        print("  Pre-computing face features ...")
        with torch.no_grad():
            loader = DataLoader(face_data, batch_size=32, shuffle=False)
            for imgs, labels in loader:
                feats = face_model.backbone(pixel_values=imgs.to(device))
                cls_tokens = feats.last_hidden_state[:, 0, :].cpu()
                for feat, lbl in zip(cls_tokens, labels.tolist()):
                    face_by_class[lbl].append(feat)

        print("  Pre-computing audio features ...")
        with torch.no_grad():
            for wav, lbl in audio_data:
                inp = wav.unsqueeze(0).to(device)
                feat = audio_model.extract_features(inp).squeeze(0).cpu()
                audio_by_class[int(lbl)].append(feat)

        # Build random same-class pairs
        self.face_feats  = []
        self.audio_feats = []
        self.labels      = []
        classes_present  = [c for c in range(NUM_CLASSES)
                            if face_by_class[c] and audio_by_class[c]]
        for _ in range(n_samples):
            c = random.choice(classes_present)
            self.face_feats.append(random.choice(face_by_class[c]))
            self.audio_feats.append(random.choice(audio_by_class[c]))
            self.labels.append(c)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.face_feats[idx], self.audio_feats[idx], self.labels[idx]


# ─── collate for variable-length waveforms ────────────────────────────────────
def _collate(batch):
    images, waveforms, labels = zip(*batch)
    max_len = max(w.shape[0] for w in waveforms)
    padded  = torch.zeros(len(waveforms), max_len)
    for i, w in enumerate(waveforms):
        padded[i, :w.shape[0]] = w
    return torch.stack(images), padded, torch.tensor(labels, dtype=torch.long)


# ─── helpers ──────────────────────────────────────────────────────────────────
def accuracy(logits, labels):
    return (logits.argmax(1) == labels).float().mean().item()


@torch.no_grad()
def evaluate(fusion_model, face_model, audio_model, loader, criterion, device,
             synthetic=False):
    fusion_model.eval()
    total_loss = total_acc = n = 0
    all_preds, all_labels = [], []

    for batch in loader:
        if synthetic:
            face_feat, audio_feat, labels = [x.to(device) for x in batch]
        else:
            images, waveforms, labels = batch
            images, waveforms, labels = (images.to(device), waveforms.to(device),
                                          labels.to(device))
            face_feat  = face_model.backbone(pixel_values=images).last_hidden_state[:, 0, :]
            audio_feat = audio_model.extract_features(waveforms)

        logits = fusion_model(face_feat, audio_feat)
        total_loss += criterion(logits, labels).item()
        total_acc  += accuracy(logits, labels)
        n += 1
        all_preds.extend(logits.argmax(1).cpu().tolist())
        all_labels.extend(labels.cpu().tolist())

    return total_loss / max(n, 1), total_acc / max(n, 1), all_preds, all_labels


# ─── CLI ─────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description="Train multimodal fusion model")
    p.add_argument("--face_ckpt",  default=None, help="Path to best_dino.pt")
    p.add_argument("--audio_ckpt", default=None, help="Path to best_audio.pt")
    p.add_argument("--dino_variant", default="vits16", choices=["vits16", "vitb16"])

    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--csv",        help="Paired CSV (image, audio, label)")
    g.add_argument("--synthetic",  action="store_true",
                   help="Use synthetic pairing from face+audio datasets")

    p.add_argument("--img_root",   default="DATASET/train",
                   help="Image root for paired CSV or face data for synthetic")
    p.add_argument("--audio_root", default="audio_data",
                   help="Audio root for paired CSV or audio data for synthetic")
    p.add_argument("--n_synthetic", type=int, default=2000,
                   help="Number of synthetic paired samples to generate")
    p.add_argument("--img_csv",    default="train_labels.csv",
                   help="Face labels CSV (for synthetic mode)")
    p.add_argument("--audio_dataset", default=None,
                   help="Audio dataset type: ravdess/cremad/folder (auto-detect)")

    p.add_argument("--epochs",     type=int,   default=20)
    p.add_argument("--batch_size", type=int,   default=32)
    p.add_argument("--lr",         type=float, default=5e-4)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--val_split",  type=float, default=0.1)
    p.add_argument("--output_dir", default="checkpoints")
    return p.parse_args()


def main():
    args   = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)

    face_dim  = 384 if args.dino_variant == "vits16" else 768
    audio_dim = 768

    print(f"\n{'='*60}")
    print(f"  Device : {device}  | Face dim : {face_dim}  | Audio dim : {audio_dim}")
    print(f"{'='*60}\n")

    # ── Load frozen feature extractors ───────────────────────────────────────
    face_model  = DINOFERModel(model_variant=args.dino_variant, freeze=True).to(device)
    audio_model = AudioFERModel(freeze_backbone=True).to(device)

    if args.face_ckpt and os.path.exists(args.face_ckpt):
        face_model.load_state_dict(torch.load(args.face_ckpt, map_location=device))
        print(f"Loaded face checkpoint : {args.face_ckpt}")
    if args.audio_ckpt and os.path.exists(args.audio_ckpt):
        audio_model.load_state_dict(torch.load(args.audio_ckpt, map_location=device))
        print(f"Loaded audio checkpoint: {args.audio_ckpt}")

    face_model.eval()
    audio_model.eval()

    # ── Build fusion model ────────────────────────────────────────────────────
    fusion = MultimodalFusionModel(face_dim=face_dim, audio_dim=audio_dim).to(device)
    print(f"Fusion trainable params: {count_trainable_params(fusion):,}\n")

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = torch.optim.AdamW(fusion.parameters(),
                                   lr=args.lr, weight_decay=args.weight_decay)

    # ── Build dataloaders ─────────────────────────────────────────────────────
    if args.synthetic:
        from dataset     import FERTrainDataset, get_val_transforms
        from audio_model import SpeechEmotionDataset

        face_raw  = FERTrainDataset(args.img_csv, args.img_root,
                                     transform=get_val_transforms())
        audio_raw = SpeechEmotionDataset(args.audio_root, dataset=args.audio_dataset)

        print("Building synthetic paired dataset...")
        paired = SyntheticPairedDataset(face_model, audio_model, face_raw, audio_raw,
                                         args.n_synthetic, device)
        n        = len(paired)
        val_n    = int(n * args.val_split)
        gen      = torch.Generator().manual_seed(42)
        idxs     = torch.randperm(n, generator=gen).tolist()
        train_ds = Subset(paired, idxs[val_n:])
        val_ds   = Subset(paired, idxs[:val_n])

        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
        val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False)
        synthetic = True
    else:
        dataset  = PairedFaceAudioDataset(args.csv, args.img_root, args.audio_root)
        n        = len(dataset)
        val_n    = int(n * args.val_split)
        gen      = torch.Generator().manual_seed(42)
        idxs     = torch.randperm(n, generator=gen).tolist()
        train_ds = Subset(dataset, idxs[val_n:])
        val_ds   = Subset(dataset, idxs[:val_n])
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                                   collate_fn=_collate)
        val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False,
                                   collate_fn=_collate)
        synthetic = False

    print(f"  Train: {len(train_ds)}   Val: {len(val_ds)}\n")

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs * len(train_loader), eta_min=1e-6)

    best_val_acc = 0.0
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    for epoch in range(1, args.epochs + 1):
        fusion.train()
        ep_loss = ep_acc = n_steps = 0
        t0 = time.time()

        for batch in train_loader:
            if synthetic:
                face_feat, audio_feat, labels = [x.to(device) for x in batch]
            else:
                images, waveforms, labels = batch
                images    = images.to(device)
                waveforms = waveforms.to(device)
                labels    = labels.to(device)
                with torch.no_grad():
                    face_feat  = face_model.backbone(pixel_values=images).last_hidden_state[:, 0, :]
                    audio_feat = audio_model.extract_features(waveforms)

            optimizer.zero_grad()
            logits = fusion(face_feat, audio_feat)
            loss   = criterion(logits, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(fusion.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            ep_loss  += loss.item()
            ep_acc   += accuracy(logits, labels)
            n_steps  += 1

        t_loss = ep_loss / max(n_steps, 1)
        t_acc  = ep_acc  / max(n_steps, 1)
        v_loss, v_acc, _, _ = evaluate(fusion, face_model, audio_model,
                                        val_loader, criterion, device, synthetic)

        print(f"Epoch {epoch:02d}/{args.epochs}  "
              f"train_loss={t_loss:.4f}  train_acc={t_acc:.4f}  "
              f"val_loss={v_loss:.4f}  val_acc={v_acc:.4f}  "
              f"({time.time()-t0:.0f}s)")

        history["train_loss"].append(t_loss)
        history["train_acc"].append(t_acc)
        history["val_loss"].append(v_loss)
        history["val_acc"].append(v_acc)

        if v_acc >= best_val_acc:
            best_val_acc = v_acc
            ckpt = os.path.join(args.output_dir, "best_fusion.pt")
            torch.save(fusion.state_dict(), ckpt)
            print(f"  [BEST] val_acc={best_val_acc:.4f} -> {ckpt}")

    results = {"val_acc": round(best_val_acc, 4), "history": history}
    rp = os.path.join(args.output_dir, "results_fusion.json")
    with open(rp, "w") as fp:
        json.dump(results, fp, indent=2)
    print(f"\n  Results -> {rp}\n  Best val acc: {best_val_acc:.4f}\n")


if __name__ == "__main__":
    main()
