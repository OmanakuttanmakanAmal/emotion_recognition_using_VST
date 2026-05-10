"""
demo_professor.py
-----------------
Comprehensive demonstration of the Multimodal Emotion Recognition system.

Sections
--------
  1. System Architecture
  2. Dataset Summary
  3. Face Model Evaluation   -- RAF-DB test set (real accuracy / F1 / confusion)
  4. Audio Model Demo        -- RAVDESS inference (emotion distribution)
  5. Multimodal Video Demo   -- RAVDESS Actor 14 MP4 (face + audio -> fused emotion)
  6. Results Summary

Run:
    python demo_professor.py
    python demo_professor.py --face_ckpt  checkpoints/best_dino.pt
    python demo_professor.py --audio_ckpt checkpoints/best_audio.pt
    python demo_professor.py --max_test   200   # faster demo (subset of test images)
"""

import os
import sys
import argparse
import time
import warnings

import torch

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

EMOTION_NAMES  = ["Surprise", "Fear", "Disgust", "Happy", "Sad", "Anger", "Neutral"]
NUM_CLASSES    = 7
SEP            = "=" * 70
SUBSEP         = "-" * 70


# =============================================================================
# Helpers
# =============================================================================

def bar(value: float, width: int = 30, fill: str = "#") -> str:
    n = int(value * width)
    return f"[{fill * n:<{width}}]"


def confusion_matrix_text(cm: list[list[int]]) -> str:
    header = "     " + "".join(f"{EMOTION_NAMES[j][:4]:>7}" for j in range(NUM_CLASSES))
    lines  = [header]
    for i in range(NUM_CLASSES):
        row_total = max(sum(cm[i]), 1)
        row = f"{EMOTION_NAMES[i][:4]:<5}" + "".join(f"{cm[i][j]:>7}" for j in range(NUM_CLASSES))
        lines.append(row)
    return "\n".join(lines)


def macro_f1_score(preds, labels) -> tuple[float, list[float]]:
    tp = [0] * NUM_CLASSES
    fp = [0] * NUM_CLASSES
    fn = [0] * NUM_CLASSES
    for p, l in zip(preds, labels):
        if p == l:
            tp[l] += 1
        else:
            fp[p] += 1
            fn[l] += 1
    f1s = []
    for c in range(NUM_CLASSES):
        denom = 2 * tp[c] + fp[c] + fn[c]
        f1s.append(2 * tp[c] / denom if denom > 0 else 0.0)
    return sum(f1s) / NUM_CLASSES, f1s


def per_class_accuracy(preds, labels) -> list[float]:
    correct = [0] * NUM_CLASSES
    total   = [0] * NUM_CLASSES
    for p, l in zip(preds, labels):
        total[l] += 1
        if p == l:
            correct[l] += 1
    return [correct[c] / max(total[c], 1) for c in range(NUM_CLASSES)]


def build_confusion_matrix(preds, labels) -> list[list[int]]:
    cm = [[0] * NUM_CLASSES for _ in range(NUM_CLASSES)]
    for p, l in zip(preds, labels):
        cm[l][p] += 1
    return cm


def header(title: str):
    print(f"\n{SEP}")
    print(f"  {title}")
    print(SEP)


def subheader(title: str):
    print(f"\n{SUBSEP}")
    print(f"  {title}")
    print(SUBSEP)


# =============================================================================
# Section 1 — Architecture
# =============================================================================

def show_architecture():
    header("SECTION 1: System Architecture")
    print("""
  MULTIMODAL EMOTION RECOGNITION SYSTEM
  Thesis: Amal Omanakuttan | TU Dublin (DIT) | 2024-2025

  Two independent signal streams are processed in parallel, then fused:

  INPUT MODALITIES
  ----------------
  1. Face Image (224 x 224 px, RAF-DB aligned)
  2. Audio Waveform (16 kHz mono, RAVDESS speech)

  VISUAL BRANCH  (Phase 1)
  -------------------------------------------------------
  Face Image (224x224 RGB)
       |
       v
  DINO ViT-S/16  [facebook/dino-vits16]
    - Architecture  : Vision Transformer, Small variant
    - Patch size    : 16 x 16 px  ->  196 patches + 1 CLS token
    - Hidden size   : 384-d
    - Pre-training  : Self-supervised DINO on ImageNet (Caron et al., ICCV 2021)
    - Head          : LayerNorm -> Linear(384,256) -> GELU -> Dropout(0.3)
                      -> Linear(256, 7)
    - Training      : Last 2 transformer blocks + head fine-tuned on RAF-DB
       |
       v
  face_feat  (1, 384)   <- CLS token from last encoder layer

  AUDIO BRANCH  (Phase 3)
  -------------------------------------------------------
  Audio Waveform (T samples @ 16 kHz)
       |
       v
  Wav2Vec2-base  [facebook/wav2vec2-base-960h]
    - Architecture  : Convolutional feature encoder + Transformer
    - Hidden size   : 768-d
    - Pre-training  : Self-supervised wav2vec 2.0 on LibriSpeech 960h
                      (Baevski et al., NeurIPS 2020)
    - Pooling       : Mean-pool all time-step hidden states
    - Head          : LayerNorm -> Linear(768,256) -> GELU -> Dropout(0.3)
                      -> Linear(256, 7)
    - Training      : Frozen backbone, head trained on RAVDESS
       |
       v
  audio_feat  (1, 768)   <- mean-pooled hidden states

  FUSION  (Phase 3)
  -------------------------------------------------------
  Concatenate: [face_feat | audio_feat]  ->  (1, 1152)
       |
       v
  MultimodalFusionModel
    - LayerNorm(1152)
    - Linear(1152 -> 512) + GELU + Dropout(0.4)
    - Linear(512  -> 128) + GELU + Dropout(0.4)
    - Linear(128  -> 7)
       |
       v
  Emotion Logits (1, 7)
  Softmax -> Predicted Emotion

  7 EMOTION CLASSES (RAF-DB taxonomy)
  ------------------------------------
  0=Surprise  1=Fear  2=Disgust  3=Happy  4=Sad  5=Anger  6=Neutral
""")


# =============================================================================
# Section 2 — Dataset Summary
# =============================================================================

def show_dataset_summary(data_root: str):
    header("SECTION 2: Dataset Summary")

    # RAF-DB face
    subheader("Face Dataset: RAF-DB (Real-world Affective Faces)")
    import csv
    csv_path = os.path.join(data_root, "train_labels.csv")
    train_counts = {}
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            lbl = int(row["label"].strip()) - 1
            train_counts[lbl] = train_counts.get(lbl, 0) + 1

    test_dir = os.path.join(data_root, "DATASET", "test")
    test_counts = {}
    for cls_folder in os.listdir(test_dir):
        cls_path = os.path.join(test_dir, cls_folder)
        if not os.path.isdir(cls_path):
            continue
        try:
            lbl = int(cls_folder) - 1
        except ValueError:
            continue
        test_counts[lbl] = sum(
            1 for f in os.listdir(cls_path)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        )

    total_train = sum(train_counts.values())
    total_test  = sum(test_counts.values())
    print(f"\n  {'Emotion':<12} {'Train':>7} {'Test':>7}  Distribution (train)")
    print(f"  {'-'*60}")
    for c in range(NUM_CLASSES):
        tr = train_counts.get(c, 0)
        te = test_counts.get(c, 0)
        pct = tr / total_train
        print(f"  {EMOTION_NAMES[c]:<12} {tr:>7} {te:>7}  {bar(pct, 25)} {pct*100:.1f}%")
    print(f"  {'-'*60}")
    print(f"  {'TOTAL':<12} {total_train:>7} {total_test:>7}")
    print(f"\n  NOTE: Class imbalance -- Happy={train_counts.get(3,0)/total_train*100:.0f}%,"
          f" Fear={train_counts.get(1,0)/total_train*100:.0f}%")
    print("  Mitigation: label smoothing (0.1) + data augmentation (ColorJitter, RandomCrop)")

    # RAVDESS audio
    subheader("Audio Dataset: RAVDESS (Speech-only)")
    audio_root = os.path.join(data_root, "audio_data")
    if os.path.isdir(audio_root):
        emo_names_ravdess = {
            1:"Neutral", 2:"Calm->Neutral", 3:"Happy", 4:"Sad",
            5:"Angry", 6:"Fearful", 7:"Disgust", 8:"Surprised"
        }
        from audio_model import _RAVDESS_MAP
        import os as _os
        counts = {}
        n_actors = set()
        for root, _, files in _os.walk(audio_root):
            for f in files:
                if not f.endswith(".wav"):
                    continue
                parts = f.replace(".wav","").split("-")
                if len(parts) != 7:
                    continue
                emo   = int(parts[2])
                actor = int(parts[6])
                n_actors.add(actor)
                raf   = _RAVDESS_MAP[emo]
                counts[raf] = counts.get(raf, 0) + 1
        total_audio = sum(counts.values())
        print(f"\n  Actors: {len(n_actors)}  |  Total files: {total_audio}")
        print(f"\n  {'Emotion':<12} {'Files':>7}  Distribution")
        print(f"  {'-'*50}")
        for c in range(NUM_CLASSES):
            n   = counts.get(c, 0)
            pct = n / max(total_audio, 1)
            print(f"  {EMOTION_NAMES[c]:<12} {n:>7}  {bar(pct, 25)} {pct*100:.1f}%")
        print(f"  {'-'*50}")
        print(f"  {'TOTAL':<12} {total_audio:>7}")
        print("\n  NOTE: Neutral includes RAVDESS 'calm' (codes 01+02 -> RAF-DB Neutral)")
        print("  RAVDESS is much more balanced than RAF-DB (good for audio branch training)")
    else:
        print("  audio_data/ not found -- skip")

    # RAVDESS video
    video_root = os.path.join(data_root, "video_data")
    if os.path.isdir(video_root):
        subheader("Video Dataset: RAVDESS Actor 14 (AV)")
        from video_utils import list_av_files
        av_files = list_av_files(video_root)
        vc = {}
        for _, lbl in av_files:
            vc[lbl] = vc.get(lbl, 0) + 1
        total_v = sum(vc.values())
        print(f"\n  Actor: 14 only  |  AV files: {total_v}")
        print(f"  (Each MP4 = 1280x720 RGB @ 30fps + 48kHz stereo audio, ~3.7s)")
        print(f"\n  {'Emotion':<12} {'Files':>7}")
        print(f"  {'-'*25}")
        for c in range(NUM_CLASSES):
            print(f"  {EMOTION_NAMES[c]:<12} {vc.get(c,0):>7}")
        print(f"  {'-'*25}")
        print(f"  {'TOTAL':<12} {total_v:>7}")


# =============================================================================
# Section 3 — Face Model Evaluation (RAF-DB test set)
# =============================================================================

def evaluate_face_model(data_root: str, face_ckpt: str | None,
                         max_samples: int | None):
    header("SECTION 3: Face Model Evaluation -- RAF-DB Test Set")

    from models  import DINOFERModel
    from dataset import build_dataloaders

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n  Device: {device}")

    subheader("Loading DINO ViT-S/16")
    model = DINOFERModel(model_variant="vits16", freeze=True).to(device)

    if face_ckpt and os.path.exists(face_ckpt):
        model.load_state_dict(torch.load(face_ckpt, map_location=device))
        print(f"  Checkpoint : {face_ckpt}")
    else:
        print("  WARNING: No checkpoint -- using random head. Predictions are meaningless.")
        print("  Train first: python src/train.py --model dino --epochs 20 --batch_size 32")

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    print(f"  Trainable params : {trainable:,} / {total:,} "
          f"({trainable/total*100:.1f}% of backbone)")

    _, _, test_loader = build_dataloaders(
        csv_path      = os.path.join(data_root, "train_labels.csv"),
        train_img_dir = os.path.join(data_root, "DATASET", "train"),
        test_dir      = os.path.join(data_root, "DATASET", "test"),
        batch_size    = 32,
        num_workers   = 0,
    )

    subheader("Running inference on test set")
    model.eval()
    all_preds, all_labels = [], []
    n_done = 0
    t0 = time.time()

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            logits = model(images)
            preds  = logits.argmax(dim=1).cpu().tolist()
            all_preds.extend(preds)
            all_labels.extend(labels.tolist())
            n_done += len(preds)
            print(f"  Processed {n_done} / {len(test_loader.dataset)} images ...", end="\r")
            if max_samples and n_done >= max_samples:
                break

    elapsed = time.time() - t0
    print(f"  Processed {n_done} images in {elapsed:.1f}s "
          f"({n_done/elapsed:.0f} img/s)           ")

    # Metrics
    correct  = sum(p == l for p, l in zip(all_preds, all_labels))
    accuracy = correct / len(all_labels)
    macro_f1, per_class_f1 = macro_f1_score(all_preds, all_labels)
    per_acc  = per_class_accuracy(all_preds, all_labels)
    cm       = build_confusion_matrix(all_preds, all_labels)

    subheader("Results")
    print(f"\n  Overall Accuracy : {accuracy*100:.2f}%   ({correct}/{len(all_labels)} correct)")
    print(f"  Macro F1 Score   : {macro_f1*100:.2f}%")

    print(f"\n  Per-class Accuracy & F1")
    print(f"  {'Emotion':<12} {'Acc':>7}  {'F1':>7}  Visual")
    print(f"  {'-'*55}")
    for c in range(NUM_CLASSES):
        acc_bar = bar(per_acc[c], 20)
        print(f"  {EMOTION_NAMES[c]:<12} {per_acc[c]*100:>6.1f}%  "
              f"{per_class_f1[c]*100:>6.1f}%  {acc_bar}")

    subheader("Confusion Matrix  (rows=true, cols=predicted)")
    print()
    print(confusion_matrix_text(cm))
    print()

    # Sample predictions
    subheader("Sample Predictions (first 10 test images)")
    sample_loader = build_dataloaders(
        csv_path      = os.path.join(data_root, "train_labels.csv"),
        train_img_dir = os.path.join(data_root, "DATASET", "train"),
        test_dir      = os.path.join(data_root, "DATASET", "test"),
        batch_size    = 10,
        num_workers   = 0,
    )[2]
    images, labels = next(iter(sample_loader))
    with torch.no_grad():
        logits = model(images.to(device))
        probs  = torch.softmax(logits, dim=-1).cpu()
        preds  = probs.argmax(dim=1)

    print(f"\n  {'#':>3}  {'True':>10}  {'Predicted':>10}  {'Conf':>6}  {'Status':>6}")
    print(f"  {'-'*50}")
    for i in range(len(labels)):
        t = EMOTION_NAMES[labels[i]]
        p = EMOTION_NAMES[preds[i]]
        c = probs[i][preds[i]].item() * 100
        ok = "OK" if preds[i] == labels[i] else "WRONG"
        print(f"  {i+1:>3}  {t:>10}  {p:>10}  {c:>5.1f}%  {ok:>6}")

    return accuracy, macro_f1


# =============================================================================
# Section 4 — Audio Model Demo (RAVDESS)
# =============================================================================

def demo_audio_model(data_root: str, audio_ckpt: str | None, n_samples: int = 20):
    header("SECTION 4: Audio Model Demo -- RAVDESS Speech")

    audio_root = os.path.join(data_root, "audio_data")
    if not os.path.isdir(audio_root):
        print("  audio_data/ not found -- skip")
        return

    from audio_model import AudioFERModel, _load_waveform

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = AudioFERModel(freeze_backbone=True).to(device)

    if audio_ckpt and os.path.exists(audio_ckpt):
        model.load_state_dict(torch.load(audio_ckpt, map_location=device))
        print(f"\n  Checkpoint: {audio_ckpt}")
    else:
        print("\n  WARNING: No audio checkpoint -- using random head.")
        print("  Train first: python src/train_audio.py --data_root audio_data/ --epochs 20")

    # Collect sample WAV files
    import random
    wav_files = []
    for root, _, files in os.walk(audio_root):
        for f in files:
            if f.endswith(".wav"):
                wav_files.append(os.path.join(root, f))
    random.seed(42)
    sample = random.sample(wav_files, min(n_samples, len(wav_files)))

    subheader(f"Inference on {len(sample)} random RAVDESS audio clips")
    model.eval()

    from audio_model import _RAVDESS_MAP
    correct = 0
    print(f"\n  {'Filename':<40} {'True':>10}  {'Predicted':>10}  {'Conf':>6}")
    print(f"  {'-'*75}")

    with torch.no_grad():
        for wav_path in sample:
            fname = os.path.basename(wav_path)
            parts = fname.replace(".wav","").split("-")
            true_raf = _RAVDESS_MAP.get(int(parts[2]))
            if true_raf is None:
                continue

            waveform = _load_waveform(wav_path, 16000).unsqueeze(0).to(device)
            logits   = model(waveform)
            probs    = torch.softmax(logits, dim=-1).squeeze(0)
            pred     = probs.argmax().item()
            conf     = probs[pred].item() * 100

            t = EMOTION_NAMES[true_raf]
            p = EMOTION_NAMES[pred]
            ok = "OK" if pred == true_raf else ""
            if ok:
                correct += 1
            print(f"  {fname:<40} {t:>10}  {p:>10}  {conf:>5.1f}%  {ok}")

    total = len(sample)
    print(f"\n  Correct: {correct}/{total}  ({correct/total*100:.1f}%)")
    print("  (Low accuracy expected without a trained checkpoint)")


# =============================================================================
# Section 5 — Multimodal Video Demo (RAVDESS Actor 14)
# =============================================================================

def demo_multimodal_video(data_root: str, face_ckpt: str | None,
                           audio_ckpt: str | None, n_clips: int = 10):
    header("SECTION 5: Multimodal Video Demo -- RAVDESS Actor 14 MP4")
    video_root = os.path.join(data_root, "video_data")
    if not os.path.isdir(video_root):
        print("  video_data/ not found -- skip")
        return

    from video_utils import list_av_files, extract_audio, extract_face_frames, \
                            frames_to_tensor
    from models      import DINOFERModel, MultimodalFusionModel
    from audio_model import AudioFERModel
    from dataset     import get_val_transforms

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load face model
    face_model = DINOFERModel(model_variant="vits16", freeze=True).to(device)
    if face_ckpt and os.path.exists(face_ckpt):
        face_model.load_state_dict(torch.load(face_ckpt, map_location=device))
    face_model.eval()

    # Load audio model
    audio_model = AudioFERModel(freeze_backbone=True).to(device)
    if audio_ckpt and os.path.exists(audio_ckpt):
        audio_model.load_state_dict(torch.load(audio_ckpt, map_location=device))
    audio_model.eval()

    # Fusion model (no checkpoint needed -- just shows the pipeline)
    fusion_model = MultimodalFusionModel(face_dim=384, audio_dim=768).to(device)
    fusion_model.eval()

    av_files = list_av_files(video_root)
    import random
    random.seed(7)
    sample = random.sample(av_files, min(n_clips, len(av_files)))
    transform = get_val_transforms(224)

    subheader(f"Running face + audio + fusion on {len(sample)} RAVDESS clips")
    print(f"\n  Pipeline per clip:")
    print(f"    MP4 -> extract middle frame -> face detect+crop -> DINO -> face_feat (384)")
    print(f"    MP4 -> extract audio track  -> resample 16kHz   -> Wav2Vec2 -> audio_feat (768)")
    print(f"    concat(face_feat, audio_feat) -> FusionMLP -> emotion")
    print()

    print(f"  {'Filename':<42} {'True':>10}  {'Face':>10}  {'Audio':>10}  {'Fused':>10}")
    print(f"  {'-'*90}")

    face_correct = audio_correct = fused_correct = total = 0

    with torch.no_grad():
        for mp4_path, true_label in sample:
            fname = os.path.basename(mp4_path)
            try:
                # --- Face branch ---
                frames = extract_face_frames(mp4_path, n_frames=1, strategy="middle")
                face_tensor = frames_to_tensor(frames, transform).to(device)
                face_feat   = face_model.backbone(
                    pixel_values=face_tensor
                ).last_hidden_state[:, 0, :]              # (1, 384)
                face_logits = face_model.head(face_feat)
                face_pred   = face_logits.argmax(dim=1).item()

                # --- Audio branch ---
                waveform    = extract_audio(mp4_path, target_sr=16000).unsqueeze(0).to(device)
                audio_feat  = audio_model.extract_features(waveform)    # (1, 768)
                audio_logits = audio_model.head(audio_feat)
                audio_pred   = audio_logits.argmax(dim=1).item()

                # --- Fusion ---
                fused_logits = fusion_model(face_feat, audio_feat)
                fused_pred   = fused_logits.argmax(dim=1).item()

                t = EMOTION_NAMES[true_label]
                f = EMOTION_NAMES[face_pred]
                a = EMOTION_NAMES[audio_pred]
                u = EMOTION_NAMES[fused_pred]

                if face_pred  == true_label: face_correct  += 1
                if audio_pred == true_label: audio_correct += 1
                if fused_pred == true_label: fused_correct += 1
                total += 1

                print(f"  {fname:<42} {t:>10}  {f:>10}  {a:>10}  {u:>10}")

            except Exception as e:
                print(f"  {fname:<42} ERROR: {e}")

    if total:
        print(f"\n  {'-'*90}")
        print(f"  {'Accuracy':<42} {'':>10}  {face_correct/total*100:>9.1f}%"
              f"  {audio_correct/total*100:>9.1f}%  {fused_correct/total*100:>9.1f}%")
        print()
        print("  NOTE: Face/audio accuracy reflect model quality (checkpoint needed for real numbers).")
        print("  The fusion model here uses a randomly-initialised head -- needs fine-tuning.")


# =============================================================================
# Section 6 — Summary
# =============================================================================

def show_summary(face_acc: float | None, face_f1: float | None):
    header("SECTION 6: Results Summary")
    print(f"""
  SYSTEM OVERVIEW
  ---------------
  Author      : Amal Omanakuttan | TU Dublin (DIT)
  Task        : 7-class emotion recognition (Surprise/Fear/Disgust/Happy/Sad/Anger/Neutral)

  DATASETS
  --------
  Face  : RAF-DB       -- 12,271 train | 3,068 test | 7 classes | real-world images
  Audio : RAVDESS      -- 1,440 files  | 24 actors  | balanced  | speech only
  Video : RAVDESS AV   -- 60 files     | 1 actor    | paired face+audio

  MODELS
  ------
  Visual  : DINO ViT-S/16 + MLP head
            Pre-trained: Self-supervised on ImageNet  (no labels)
            Fine-tuned : Last 2 transformer blocks on RAF-DB
            Trainable  : ~3.6M parameters

  Audio   : Wav2Vec2-base + MLP head
            Pre-trained: ASR on LibriSpeech 960h
            Fine-tuned : Head only on RAVDESS (backbone frozen)
            Trainable  : ~0.4M parameters

  Fusion  : Late-fusion MLP (concat 384+768 = 1152-d -> 512 -> 128 -> 7)
            Trained on paired or synthetic A/V data
            Trainable  : ~0.7M parameters

  FACE MODEL RESULTS (RAF-DB test set)
  -------------------------------------""")

    if face_acc is not None:
        print(f"  Accuracy  : {face_acc*100:.2f}%")
        print(f"  Macro F1  : {face_f1*100:.2f}%")
        print(f"  Baseline  : Random chance = 14.3%  |  Majority-class = ~38.9% (Happy)")
    else:
        print("  Not evaluated (no checkpoint found)")

    print(f"""
  TRAINING ROADMAP
  ----------------
  Phase 1 (done): Face model -- train.py  -- DINO on RAF-DB
  Phase 3a (next): Audio model -- train_audio.py -- Wav2Vec2 on RAVDESS
  Phase 3b (next): Fusion model -- train_multimodal.py -- on paired data

  REFERENCES
  ----------
  DINO      : Caron et al., ICCV 2021 -- Self-supervised ViT
  ConvNeXt  : Liu et al., CVPR 2022
  Wav2Vec2  : Baevski et al., NeurIPS 2020
  RAF-DB    : Li et al., CVPR 2017
  RAVDESS   : Livingstone & Russo, PLOS ONE 2018
""")


# =============================================================================
# Main
# =============================================================================

def parse_args():
    p = argparse.ArgumentParser(description="Professor demonstration script")
    p.add_argument("--data_root",   default=".",
                   help="Project root (default: current dir)")
    p.add_argument("--face_ckpt",   default=None,
                   help="Path to best_dino.pt (auto-detected if omitted)")
    p.add_argument("--audio_ckpt",  default=None,
                   help="Path to best_audio.pt (auto-detected if omitted)")
    p.add_argument("--max_test",    type=int, default=None,
                   help="Limit RAF-DB test images (e.g. 200 for a quick demo)")
    p.add_argument("--n_audio",     type=int, default=20,
                   help="Number of RAVDESS audio clips to demo (default 20)")
    p.add_argument("--n_video",     type=int, default=10,
                   help="Number of RAVDESS video clips to demo (default 10)")
    p.add_argument("--skip_face",   action="store_true")
    p.add_argument("--skip_audio",  action="store_true")
    p.add_argument("--skip_video",  action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    dr   = args.data_root

    # Auto-detect checkpoints
    if args.face_ckpt is None:
        default = os.path.join(dr, "checkpoints", "best_dino.pt")
        if os.path.exists(default):
            args.face_ckpt = default
    if args.audio_ckpt is None:
        default = os.path.join(dr, "checkpoints", "best_audio.pt")
        if os.path.exists(default):
            args.audio_ckpt = default

    print(SEP)
    print("  MULTIMODAL EMOTION RECOGNITION -- PROFESSOR DEMONSTRATION")
    print("  Amal Omanakuttan | TU Dublin (DIT) | 2024-2025")
    print(SEP)
    print(f"\n  Face checkpoint  : {args.face_ckpt or 'none (will use random head)'}")
    print(f"  Audio checkpoint : {args.audio_ckpt or 'none (will use random head)'}")

    show_architecture()
    show_dataset_summary(dr)

    face_acc = face_f1 = None
    if not args.skip_face:
        face_acc, face_f1 = evaluate_face_model(dr, args.face_ckpt, args.max_test)

    if not args.skip_audio:
        demo_audio_model(dr, args.audio_ckpt, n_samples=args.n_audio)

    if not args.skip_video:
        demo_multimodal_video(dr, args.face_ckpt, args.audio_ckpt, n_clips=args.n_video)

    show_summary(face_acc, face_f1)


if __name__ == "__main__":
    main()
