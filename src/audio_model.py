"""
audio_model.py
--------------
Audio emotion recognition using HuggingFace Wav2Vec2.

Phase 3 — Voice Fusion:
    • AudioFERModel: Wav2Vec2 backbone + MLP head (standalone audio classifier).
    • SpeechEmotionDataset: supports RAVDESS and CREMA-D out of the box.
    • load_audio_for_inference: helper for single-file inference.

Label mapping (0-indexed, matches RAF-DB face labels):
    0=Surprise, 1=Fear, 2=Disgust, 3=Happy, 4=Sad, 5=Anger, 6=Neutral
"""

import os
import re
import torch
import torch.nn as nn
from torch.utils.data import Dataset

try:
    from transformers import Wav2Vec2Processor, Wav2Vec2Model
    _WAV2VEC_AVAILABLE = True
except ImportError:
    _WAV2VEC_AVAILABLE = False

NUM_CLASSES = 7
WAV2VEC_HIDDEN_DIM = 768   # wav2vec2-base hidden size

EMOTION_NAMES = ["Surprise", "Fear", "Disgust", "Happy", "Sad", "Anger", "Neutral"]

# ── RAVDESS: emotion code (1-indexed) → RAF-DB label (0-indexed) ─────────────
# RAVDESS codes: 01=neutral, 02=calm, 03=happy, 04=sad,
#                05=angry, 06=fearful, 07=disgust, 08=surprised
_RAVDESS_MAP = {1: 6, 2: 6, 3: 3, 4: 4, 5: 5, 6: 1, 7: 2, 8: 0}

# ── CREMA-D: emotion string → RAF-DB label (0-indexed) ───────────────────────
_CREMAD_MAP = {"ANG": 5, "DIS": 2, "FEA": 1, "HAP": 3, "NEU": 6, "SAD": 4}


# ─── AudioFERModel ─────────────────────────────────────────────────────────────
class AudioFERModel(nn.Module):
    """
    Wav2Vec2 backbone (frozen by default) + MLP head.

    Input  : raw waveform, shape (B, T), 16 kHz sampling rate.
    Output : logits (B, NUM_CLASSES).

    extract_features() returns the mean-pooled (B, 768) embedding for fusion.
    """

    MODEL_ID = "facebook/wav2vec2-base-960h"

    def __init__(self, num_classes: int = NUM_CLASSES,
                 freeze_backbone: bool = True, dropout: float = 0.3):
        super().__init__()
        if not _WAV2VEC_AVAILABLE:
            raise RuntimeError(
                "transformers not installed. Run: pip install transformers torchaudio soundfile"
            )
        self.processor = Wav2Vec2Processor.from_pretrained(self.MODEL_ID)
        self.backbone  = Wav2Vec2Model.from_pretrained(self.MODEL_ID)

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        self.head = nn.Sequential(
            nn.LayerNorm(WAV2VEC_HIDDEN_DIM),
            nn.Linear(WAV2VEC_HIDDEN_DIM, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

    def extract_features(self, input_values: torch.Tensor) -> torch.Tensor:
        """Mean-pooled audio embedding (B, 768) for late fusion."""
        # Normalize per-sample (mimics Wav2Vec2Processor normalization)
        mean = input_values.mean(dim=-1, keepdim=True)
        std  = input_values.std(dim=-1, keepdim=True).clamp(min=1e-8)
        input_values = (input_values - mean) / std
        outputs = self.backbone(input_values=input_values)
        return outputs.last_hidden_state.mean(dim=1)

    def forward(self, input_values: torch.Tensor) -> torch.Tensor:
        return self.head(self.extract_features(input_values))

    @property
    def processor_ref(self):
        return self.processor


# ─── SpeechEmotionDataset ──────────────────────────────────────────────────────
class SpeechEmotionDataset(Dataset):
    """
    Unified dataset for RAVDESS and CREMA-D audio emotion corpora.

    Automatically detects dataset type from filename patterns.

    RAVDESS layout (nested Actor folders):
        ravdess_root/
            Actor_01/
                03-01-05-01-02-01-12.wav   ← 3rd field = emotion code
            Actor_02/
                ...

    CREMA-D layout (flat folder):
        cremad_root/
            1001_DFA_ANG_XX.wav            ← 3rd underscore-field = emotion code

    Folder-organised layout (one subfolder per RAF-DB class 0-6):
        audio_root/
            0/  *.wav      ← Surprise
            1/  *.wav      ← Fear
            ...
            6/  *.wav      ← Neutral

    Args:
        root_dir  : path to dataset root.
        dataset   : "ravdess" | "cremad" | "folder" (auto-detected if None).
        transform : optional callable applied to the raw waveform numpy array.
        sample_rate: expected sample rate; files are resampled on load if needed.
    """

    def __init__(self, root_dir: str, dataset: str | None = None,
                 sample_rate: int = 16000):
        self.root_dir    = root_dir
        self.sample_rate = sample_rate
        self.samples: list[tuple[str, int]] = []  # (wav_path, label_0indexed)

        kind = dataset or self._detect(root_dir)
        if kind == "ravdess":
            self._load_ravdess(root_dir)
        elif kind == "cremad":
            self._load_cremad(root_dir)
        elif kind == "folder":
            self._load_folder(root_dir)
        else:
            raise ValueError(f"Unknown dataset type '{kind}'. Use ravdess/cremad/folder.")

    # ── auto-detection ────────────────────────────────────────────────────────
    @staticmethod
    def _detect(root: str) -> str:
        for entry in os.listdir(root):
            if entry.startswith("Actor_"):
                return "ravdess"
            if re.match(r"^\d{4}_[A-Z]+_[A-Z]+", entry):
                return "cremad"
            if entry.isdigit():
                return "folder"
        # fall back: look one level deeper
        for entry in os.listdir(root):
            sub = os.path.join(root, entry)
            if os.path.isdir(sub):
                for f in os.listdir(sub):
                    if re.match(r"^\d{2}-\d{2}-\d{2}", f):
                        return "ravdess"
                    if re.match(r"^\d{4}_[A-Z]+_[A-Z]+", f):
                        return "cremad"
        return "folder"

    # ── RAVDESS loader ────────────────────────────────────────────────────────
    def _load_ravdess(self, root: str):
        for actor_dir in sorted(os.listdir(root)):
            actor_path = os.path.join(root, actor_dir)
            if not os.path.isdir(actor_path):
                continue
            for fname in os.listdir(actor_path):
                if not fname.endswith(".wav"):
                    continue
                parts = fname.replace(".wav", "").split("-")
                if len(parts) < 3:
                    continue
                try:
                    emo_code = int(parts[2])
                except ValueError:
                    continue
                label = _RAVDESS_MAP.get(emo_code)
                if label is None:
                    continue
                self.samples.append((os.path.join(actor_path, fname), label))

    # ── CREMA-D loader ────────────────────────────────────────────────────────
    def _load_cremad(self, root: str):
        for fname in sorted(os.listdir(root)):
            if not fname.endswith(".wav"):
                continue
            parts = fname.replace(".wav", "").split("_")
            if len(parts) < 3:
                continue
            emo_str = parts[2].upper()
            label   = _CREMAD_MAP.get(emo_str)
            if label is None:
                continue
            self.samples.append((os.path.join(root, fname), label))

    # ── folder-organised loader ───────────────────────────────────────────────
    def _load_folder(self, root: str):
        for class_folder in sorted(os.listdir(root)):
            class_path = os.path.join(root, class_folder)
            if not os.path.isdir(class_path):
                continue
            try:
                label = int(class_folder)
            except ValueError:
                continue
            if not (0 <= label < NUM_CLASSES):
                continue
            for fname in os.listdir(class_path):
                if fname.lower().endswith(".wav"):
                    self.samples.append((os.path.join(class_path, fname), label))

    # ── PyTorch Dataset interface ─────────────────────────────────────────────
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        wav_path, label = self.samples[idx]
        waveform = _load_waveform(wav_path, self.sample_rate)
        return waveform, label

    def class_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {n: 0 for n in EMOTION_NAMES}
        for _, lbl in self.samples:
            counts[EMOTION_NAMES[lbl]] += 1
        return counts


# ─── waveform loading helper ──────────────────────────────────────────────────

def _load_waveform(wav_path: str, target_sr: int = 16000) -> torch.Tensor:
    """
    Load a .wav file, resample to target_sr, convert to mono.
    Returns a 1-D float32 tensor of shape (T,).
    """
    # Use soundfile directly -- torchaudio.load in >=2.11 requires torchcodec for WAV
    try:
        import soundfile as sf
        import numpy as np
        data, sr = sf.read(wav_path, dtype="float32", always_2d=True)  # (T, channels)
        waveform  = torch.from_numpy(data.T)   # (channels, T)
    except Exception:
        try:
            import torchaudio
            waveform, sr = torchaudio.load(wav_path)
        except Exception as e:
            raise RuntimeError(f"Cannot load {wav_path}: {e}")

    waveform = waveform.mean(dim=0)   # mono: (T,)
    if sr != target_sr:
        import torchaudio
        waveform = torchaudio.functional.resample(waveform, sr, target_sr)
    return waveform


def load_audio_for_inference(wav_path: str, processor, device: torch.device):
    """
    Load a single .wav file and return processor output ready for AudioFERModel.
    """
    waveform = _load_waveform(wav_path).numpy()
    inputs   = processor(waveform, sampling_rate=16000,
                          return_tensors="pt", padding=True)
    return inputs.input_values.to(device)


# ─── collate function for DataLoader ─────────────────────────────────────────

def audio_collate_fn(batch):
    """
    Pad variable-length waveforms in a batch to the same length.
    Returns (padded_tensor [B, T_max], labels [B]).
    """
    waveforms, labels = zip(*batch)
    max_len = max(w.shape[0] for w in waveforms)
    padded  = torch.zeros(len(waveforms), max_len)
    for i, w in enumerate(waveforms):
        padded[i, :w.shape[0]] = w
    return padded, torch.tensor(labels, dtype=torch.long)


def build_audio_dataloaders(root_dir: str, dataset: str | None = None,
                             batch_size: int = 16, val_split: float = 0.1,
                             num_workers: int = 0, seed: int = 42):
    """
    Returns train_loader, val_loader for an audio emotion dataset.
    """
    from torch.utils.data import Subset
    full = SpeechEmotionDataset(root_dir, dataset=dataset)
    n         = len(full)
    val_size  = int(n * val_split)
    generator = torch.Generator().manual_seed(seed)
    indices   = torch.randperm(n, generator=generator).tolist()

    train_ds = Subset(full, indices[val_size:])
    val_ds   = Subset(full, indices[:val_size])

    from torch.utils.data import DataLoader
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, collate_fn=audio_collate_fn)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, collate_fn=audio_collate_fn)

    counts = full.class_counts()
    print(f"  Audio train samples : {len(train_ds)}")
    print(f"  Audio val   samples : {len(val_ds)}")
    print(f"  Class distribution  : { {k: v for k, v in counts.items() if v > 0} }")
    return train_loader, val_loader
