"""
models.py
---------
Two vision models for Facial Emotion Recognition:

1.  ConvNeXtFERModel  – wraps the HuggingFace DrGM ConvNeXt-V2L already
                        fine-tuned for 7-class FER. We replace the final head
                        for fine-tuning on our dataset.

2.  DINOFERModel      – wraps facebook/dino-vitb16 (or vits16). DINO is a
                        self-supervised ViT; we freeze the backbone and train
                        only a lightweight MLP classification head.
                        For full fine-tuning (Colab GPU), set freeze=False.
"""

import torch
import torch.nn as nn
from transformers import (
    AutoImageProcessor,
    AutoModelForImageClassification,
    ViTModel,
    ViTImageProcessor,
)

NUM_CLASSES = 7


# ─── Helper: MLP head ────────────────────────────────────────────────────────
class MLPHead(nn.Module):
    """
    Two-layer MLP classifier with dropout and GELU activation.
    Suitable for attaching on top of frozen ViT/CNN features.
    """

    def __init__(self, in_features: int, hidden: int, num_classes: int,
                 dropout: float = 0.3):
        super().__init__()
        self.head = nn.Sequential(
            nn.LayerNorm(in_features),
            nn.Linear(in_features, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, num_classes),
        )

    def forward(self, x):
        return self.head(x)


# ─── 1. ConvNeXt FER Model ───────────────────────────────────────────────────
class ConvNeXtFERModel(nn.Module):
    """
    Fine-tunable ConvNeXt-V2L from HuggingFace.

    Strategy:
      • Load DrGM's model (already trained on 7-class FER).
      • Replace the classifier head to match our label ordering.
      • Optionally freeze the backbone to enable fast CPU/low-GPU training.
    """

    MODEL_ID = "DrGM/DrGM-ConvNeXt-V2L-Facial-Emotion-Recognition"

    def __init__(self, num_classes: int = NUM_CLASSES, freeze_backbone: bool = False):
        super().__init__()
        self.processor = AutoImageProcessor.from_pretrained(self.MODEL_ID)
        self.backbone = AutoModelForImageClassification.from_pretrained(self.MODEL_ID)

        # Identify the in_features of the existing classifier
        in_features = self.backbone.classifier.in_features  # type: ignore[union-attr]

        # Replace classifier head
        self.backbone.classifier = MLPHead(in_features, 512, num_classes)

        if freeze_backbone:
            for name, param in self.backbone.named_parameters():
                if "classifier" not in name:
                    param.requires_grad = False

    def forward(self, pixel_values):
        outputs = self.backbone(pixel_values=pixel_values)
        return outputs.logits

    @property
    def processor_ref(self):
        return self.processor


# ─── 2. DINO ViT FER Model ───────────────────────────────────────────────────
class DINOFERModel(nn.Module):
    """
    Vision-Transformer feature extractor (DINO-pretrained) + MLP head.

    DINO backbone is self-supervised; we attach an emotion classifier on top.

    Args:
        model_variant: 'vits16' (6M params, fast) or 'vitb16' (86M params, better).
        freeze:         If True, only the MLP head is trained (CPU/low-GPU friendly).
                        Set False for full fine-tuning on Colab.
        unfreeze_last_n_blocks: If freeze=True, still unfreeze the last N transformer
                        blocks for richer gradient signal.
    """

    VARIANTS = {
        "vits16": ("facebook/dino-vits16", 384),
        "vitb16": ("facebook/dino-vitb16", 768),
    }

    def __init__(
        self,
        model_variant: str = "vits16",
        num_classes: int = NUM_CLASSES,
        freeze: bool = True,
        unfreeze_last_n_blocks: int = 2,
        dropout: float = 0.3,
    ):
        super().__init__()
        if model_variant not in self.VARIANTS:
            raise ValueError(f"model_variant must be one of {list(self.VARIANTS)}")

        model_id, hidden_size = self.VARIANTS[model_variant]
        self.processor = ViTImageProcessor.from_pretrained(model_id)
        self.backbone = ViTModel.from_pretrained(model_id)
        self.head = MLPHead(hidden_size, 256, num_classes, dropout=dropout)

        if freeze:
            # Freeze the entire backbone first
            for param in self.backbone.parameters():
                param.requires_grad = False
            # Selectively unfreeze the last N encoder blocks
            num_blocks = len(self.backbone.encoder.layer)
            for i in range(num_blocks - unfreeze_last_n_blocks, num_blocks):
                for param in self.backbone.encoder.layer[i].parameters():
                    param.requires_grad = True
            # Always keep LayerNorm trainable
            for param in self.backbone.layernorm.parameters():
                param.requires_grad = True

    def forward(self, pixel_values):
        outputs = self.backbone(pixel_values=pixel_values)
        # CLS token embedding → classification head
        cls_token = outputs.last_hidden_state[:, 0, :]
        return self.head(cls_token)

    @property
    def processor_ref(self):
        return self.processor


# ─── 3. Dual-Feature Fusion Model (Face + Audio stubs) ───────────────────────
class MultimodalFusionModel(nn.Module):
    """
    Late-fusion model that concatenates face and audio embeddings,
    then passes them through an MLP to predict emotion.

    face_dim  : output dimension of the visual model (384 for vits16, 768 for vitb16)
    audio_dim : output dimension of the audio model
                (e.g. 768 for wav2vec2-base, 1024 for wav2vec2-large)
    """

    def __init__(
        self,
        face_dim: int = 384,
        audio_dim: int = 768,
        num_classes: int = NUM_CLASSES,
        dropout: float = 0.4,
    ):
        super().__init__()
        fused_dim = face_dim + audio_dim
        self.classifier = nn.Sequential(
            nn.LayerNorm(fused_dim),
            nn.Linear(fused_dim, 512),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(512, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, face_feat: torch.Tensor, audio_feat: torch.Tensor):
        """
        Args:
            face_feat  : (B, face_dim)
            audio_feat : (B, audio_dim)
        Returns:
            logits     : (B, num_classes)
        """
        combined = torch.cat([face_feat, audio_feat], dim=-1)
        return self.classifier(combined)


def count_trainable_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
