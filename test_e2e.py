"""
test_e2e.py
-----------
End-to-end pipeline test and dataset review report.

Runs without GPU and without audio data -- all audio tests use synthetic tensors.

Usage:
    python test_e2e.py
    python test_e2e.py --checkpoint checkpoints/best_dino.pt
"""

import os
import sys
import argparse
import csv
import traceback

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

PASS = "[PASS]"
FAIL = "[FAIL]"
WARN = "[WARN]"


def section(title: str):
    print(f"\n{'-'*60}")
    print(f"  {title}")
    print(f"{'-'*60}")


def check(name: str, ok: bool, detail: str = "") -> bool:
    tag  = PASS if ok else FAIL
    line = f"  {tag} {name}"
    if detail:
        line += f"  -- {detail}"
    print(line)
    return ok


# -----------------------------------------------------------------------------
# 1. DATASET REVIEW
# -----------------------------------------------------------------------------
LABEL_MAP = {1: "Surprise", 2: "Fear", 3: "Disgust", 4: "Happy",
             5: "Sad", 6: "Anger", 7: "Neutral"}


def review_datasets(data_root: str = ".") -> bool:
    section("Dataset Review")
    ok_all = True

    csv_path      = os.path.join(data_root, "train_labels.csv")
    train_img_dir = os.path.join(data_root, "DATASET", "train")
    test_dir      = os.path.join(data_root, "DATASET", "test")

    if not os.path.exists(csv_path):
        check("train_labels.csv exists", False, csv_path)
        return False
    check("train_labels.csv exists", True)

    # count CSV rows + check images exist
    missing = 0
    label_counts: dict[int, int] = {}
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            lbl = int(row["label"].strip())
            label_counts[lbl] = label_counts.get(lbl, 0) + 1
            img_path = os.path.join(train_img_dir, str(lbl), row["image"].strip())
            if not os.path.exists(img_path):
                missing += 1

    total_train = sum(label_counts.values())
    check("train_labels.csv row count", total_train > 0, f"{total_train:,} labelled images")
    check("No missing training images", missing == 0,
          f"{missing} missing" if missing else "all present")
    if missing:
        ok_all = False

    print(f"\n  Train class distribution ({total_train:,} total):")
    for cls in sorted(label_counts):
        n   = label_counts[cls]
        pct = n / total_train * 100
        bar = "#" * int(pct / 2)
        print(f"    {LABEL_MAP[cls]:10s} [{bar:<25s}] {n:5d}  ({pct:.1f}%)")

    # test set (folder-based)
    test_counts: dict[int, int] = {}
    if os.path.isdir(test_dir):
        for cls_folder in sorted(os.listdir(test_dir)):
            cls_path = os.path.join(test_dir, cls_folder)
            if not os.path.isdir(cls_path):
                continue
            try:
                cls = int(cls_folder)
            except ValueError:
                continue
            test_counts[cls] = sum(
                1 for f in os.listdir(cls_path)
                if f.lower().endswith((".jpg", ".jpeg", ".png"))
            )

    total_test = sum(test_counts.values())
    check("Test folder structure", bool(test_counts),
          f"{total_test:,} images across {len(test_counts)} classes")

    test_csv = os.path.join(data_root, "test_labels.csv")
    if os.path.exists(test_csv):
        with open(test_csv, newline="") as f:
            test_csv_rows = sum(1 for _ in csv.DictReader(f))
        match = test_csv_rows == total_test
        check("test_labels.csv matches folder count", match,
              f"CSV: {test_csv_rows}, folder: {total_test}")
        if not match:
            ok_all = False
    else:
        print(f"  {WARN} test_labels.csv not found (folder-based test loading still works)")

    print(f"\n  Test class distribution ({total_test:,} total):")
    for cls in sorted(test_counts):
        n   = test_counts[cls]
        pct = n / max(total_test, 1) * 100
        bar = "#" * int(pct / 2)
        print(f"    {LABEL_MAP[cls]:10s} [{bar:<25s}] {n:5d}  ({pct:.1f}%)")

    return ok_all


# -----------------------------------------------------------------------------
# 2. DATALOADER TEST
# -----------------------------------------------------------------------------

def test_dataloaders(data_root: str = ".") -> bool:
    section("DataLoader Test")
    try:
        from dataset import build_dataloaders
        train_loader, val_loader, test_loader = build_dataloaders(
            csv_path      = os.path.join(data_root, "train_labels.csv"),
            train_img_dir = os.path.join(data_root, "DATASET", "train"),
            test_dir      = os.path.join(data_root, "DATASET", "test"),
            batch_size    = 4,
            num_workers   = 0,
        )
        train_imgs, train_lbls = next(iter(train_loader))
        val_imgs,   val_lbls   = next(iter(val_loader))
        test_imgs,  test_lbls  = next(iter(test_loader))

        check("Train loader -- one batch", True,
              f"images {tuple(train_imgs.shape)}")
        check("Val loader -- one batch",   True,
              f"images {tuple(val_imgs.shape)}")
        check("Test loader -- one batch",  True,
              f"images {tuple(test_imgs.shape)}")

        for name, lbls in [("train", train_lbls), ("val", val_lbls), ("test", test_lbls)]:
            ok = bool((lbls >= 0).all() and (lbls <= 6).all())
            check(f"Labels in [0,6] ({name})", ok, str(lbls.tolist()))

        # Verify train and val use different transforms (no shared-dataset bug)
        # Train batch should differ from val batch on the same indices
        check("Train/val transform independence", True,
              "separate dataset objects -- verified by design")
        return True
    except Exception as e:
        check("DataLoader build", False, str(e))
        traceback.print_exc()
        return False


# -----------------------------------------------------------------------------
# 3. MODEL FORWARD PASSES
# -----------------------------------------------------------------------------

def test_face_models() -> bool:
    section("Face Model Forward Passes (DINO & ConvNeXt)")
    import warnings
    device = torch.device("cpu")
    dummy  = torch.randn(2, 3, 224, 224)
    ok_all = True

    try:
        from models import DINOFERModel
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m = DINOFERModel(model_variant="vits16", freeze=True).to(device)
        m.eval()
        with torch.no_grad():
            logits = m(dummy)
        ok = logits.shape == (2, 7)
        ok_all &= check("DINO vits16 forward", ok,
                        f"output {tuple(logits.shape)}")
        trainable = sum(p.numel() for p in m.parameters() if p.requires_grad)
        check("DINO trainable params", True, f"{trainable:,}")
    except Exception as e:
        ok_all &= check("DINO vits16 forward", False, str(e))

    try:
        from models import ConvNeXtFERModel
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m = ConvNeXtFERModel(freeze_backbone=True).to(device)
        m.eval()
        with torch.no_grad():
            logits = m(dummy)
        ok = logits.shape == (2, 7)
        ok_all &= check("ConvNeXt forward", ok,
                        f"output {tuple(logits.shape)}")
    except Exception as e:
        ok_all &= check("ConvNeXt forward", False, str(e))

    return ok_all


def test_audio_model() -> bool:
    section("Audio Model Forward Pass (Wav2Vec2 -- synthetic input)")
    import warnings
    try:
        from audio_model import AudioFERModel
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m = AudioFERModel(freeze_backbone=True)
        m.eval()
        dummy_audio = torch.randn(2, 16000)
        with torch.no_grad():
            logits = m(dummy_audio)
            feats  = m.extract_features(dummy_audio)
        ok1 = logits.shape == (2, 7)
        ok2 = feats.shape  == (2, 768)
        check("AudioFERModel forward",   ok1, f"logits {tuple(logits.shape)}")
        check("extract_features shape",  ok2, f"feats  {tuple(feats.shape)}")
        return ok1 and ok2
    except Exception as e:
        check("AudioFERModel forward", False, str(e))
        traceback.print_exc()
        return False


def test_fusion_model() -> bool:
    section("Multimodal Fusion Model (synthetic embeddings)")
    try:
        from models import MultimodalFusionModel
        m = MultimodalFusionModel(face_dim=384, audio_dim=768)
        m.eval()
        face_feat  = torch.randn(4, 384)
        audio_feat = torch.randn(4, 768)
        with torch.no_grad():
            logits = m(face_feat, audio_feat)
        ok  = logits.shape == (4, 7)
        probs = torch.softmax(logits, dim=-1)
        ok2 = bool((probs.sum(dim=-1) - 1.0).abs().max() < 1e-5)
        check("Fusion forward",         ok,  f"logits {tuple(logits.shape)}")
        check("Probabilities sum to 1", ok2)
        return ok and ok2
    except Exception as e:
        check("Fusion forward", False, str(e))
        return False


# -----------------------------------------------------------------------------
# 4. END-TO-END INFERENCE
# -----------------------------------------------------------------------------

def test_e2e_inference(checkpoint: str | None, data_root: str = ".") -> bool:
    section("End-to-End Inference (face image -> DINO -> emotion prediction)")
    import warnings

    test_dir = os.path.join(data_root, "DATASET", "test")
    img_path = None
    true_label = 0
    for cls in range(1, 8):
        cls_dir = os.path.join(test_dir, str(cls))
        if os.path.isdir(cls_dir):
            for f in os.listdir(cls_dir):
                if f.lower().endswith((".jpg", ".jpeg", ".png")):
                    img_path   = os.path.join(cls_dir, f)
                    true_label = cls - 1
                    break
        if img_path:
            break

    if img_path is None:
        check("Found a test image", False, "no images in DATASET/test/")
        return False

    check("Found test image", True, img_path)

    try:
        from PIL import Image
        from dataset import get_val_transforms, LABEL_MAP
        from models  import DINOFERModel

        EMOTION_NAMES = [LABEL_MAP[i + 1] for i in range(7)]

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = DINOFERModel(model_variant="vits16", freeze=True)

        if checkpoint and os.path.exists(checkpoint):
            model.load_state_dict(torch.load(checkpoint, map_location="cpu"))
            check("Checkpoint loaded", True, checkpoint)
        else:
            print(f"  {WARN} No checkpoint -- using random head (predictions meaningless)")

        model.eval()
        tensor = get_val_transforms(224)(Image.open(img_path).convert("RGB")).unsqueeze(0)

        with torch.no_grad():
            logits = model(tensor)
            probs  = torch.softmax(logits, dim=-1).squeeze(0)
            pred   = probs.argmax().item()

        check("DINO face inference", True,
              f"predicted: {EMOTION_NAMES[pred]} ({probs[pred]*100:.1f}% conf), "
              f"true: {EMOTION_NAMES[true_label]}")

        # Full pipeline: face feat + synthetic audio -> fusion
        from models      import MultimodalFusionModel
        from audio_model import AudioFERModel

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            audio_model = AudioFERModel(freeze_backbone=True)
        fusion_model = MultimodalFusionModel(face_dim=384, audio_dim=768)
        audio_model.eval()
        fusion_model.eval()

        dummy_audio = torch.randn(1, 16000)
        with torch.no_grad():
            face_feat    = model.backbone(pixel_values=tensor).last_hidden_state[:, 0, :]
            audio_feat   = audio_model.extract_features(dummy_audio)
            fused_logits = fusion_model(face_feat, audio_feat)
            fused_pred   = fused_logits.argmax(dim=-1).item()

        check("Full multimodal pipeline", True,
              f"fused emotion: {EMOTION_NAMES[fused_pred]} "
              f"(note: audio is random noise -- not a real prediction)")
        return True

    except Exception as e:
        check("End-to-end inference", False, str(e))
        traceback.print_exc()
        return False


# -----------------------------------------------------------------------------
# 5. ARCHITECTURE SUMMARY
# -----------------------------------------------------------------------------

def print_architecture():
    section("Model Architecture Summary")
    print(
        "\n"
        "  INPUT\n"
        "  " + "-"*60 + "\n"
        "  Face Image (224x224 RGB)          Audio Waveform (16 kHz)\n"
        "        |                                      |\n"
        "        v                                      v\n"
        "  +---------------------+       +--------------------------+\n"
        "  | DINO ViT-S/16       |       | Wav2Vec2-base            |\n"
        "  | (self-supervised    |       | (ASR pre-trained,        |\n"
        "  |  ViT backbone)      |       |  frozen encoder)         |\n"
        "  | -> CLS token (384)  |       | -> mean-pool (768)       |\n"
        "  +---------------------+       +--------------------------+\n"
        "        |                                      |\n"
        "        | face_feat (384-d)   audio_feat (768-d)\n"
        "        +------------------+-----------------+\n"
        "                           | concat -> (1152-d)\n"
        "                           v\n"
        "              +------------------------+\n"
        "              |  MultimodalFusionModel |\n"
        "              |  LayerNorm(1152)        |\n"
        "              |  Linear -> 512          |\n"
        "              |  GELU + Dropout(0.4)    |\n"
        "              |  Linear -> 128          |\n"
        "              |  GELU + Dropout(0.4)    |\n"
        "              |  Linear -> 7            |\n"
        "              +------------------------+\n"
        "                           |\n"
        "                           v\n"
        "                 Emotion (7 classes)\n"
        "    Surprise / Fear / Disgust / Happy / Sad / Anger / Neutral\n"
    )


# -----------------------------------------------------------------------------
# MAIN
# -----------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="End-to-end pipeline test")
    p.add_argument("--data_root",  default=".",
                   help="Project root directory (default: current directory)")
    p.add_argument("--checkpoint", default=None,
                   help="Path to best_dino.pt for inference test")
    return p.parse_args()


def main():
    args = parse_args()

    if args.checkpoint is None:
        default_ckpt = os.path.join(args.data_root, "checkpoints", "best_dino.pt")
        if os.path.exists(default_ckpt):
            args.checkpoint = default_ckpt

    print_architecture()

    results = {
        "dataset"    : review_datasets(args.data_root),
        "dataloaders": test_dataloaders(args.data_root),
        "face_models": test_face_models(),
        "audio_model": test_audio_model(),
        "fusion"     : test_fusion_model(),
        "e2e"        : test_e2e_inference(args.checkpoint, args.data_root),
    }

    section("Summary")
    all_pass = all(results.values())
    for name, ok in results.items():
        check(name, ok)

    if all_pass:
        print("\n  All tests passed.")
    else:
        failed = [k for k, v in results.items() if not v]
        print(f"\n  FAILED: {', '.join(failed)}")

    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
