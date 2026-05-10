"""
predict.py
----------
Run inference on a single image (or batch of images) using a saved checkpoint.

Usage:
    # With DINO model
    python predict.py --model dino --image path/to/face.jpg \
                      --checkpoint checkpoints/best_dino.pt

    # With ConvNeXt model
    python predict.py --model convnext --image path/to/face.jpg \
                      --checkpoint checkpoints/best_convnext.pt

    # Batch predict on a folder
    python predict.py --model dino --image_dir DATASET/test/1 \
                      --checkpoint checkpoints/best_dino.pt
"""

import os
import sys
import argparse
import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(__file__))
from dataset import get_val_transforms, LABEL_MAP
from models  import ConvNeXtFERModel, DINOFERModel

EMOTION_NAMES = [LABEL_MAP[i + 1] for i in range(7)]
EMOTION_EMOJI = ["😲", "😨", "🤢", "😄", "😢", "😡", "😐"]


def load_model(model_name: str, checkpoint: str | None, device: torch.device):
    if model_name == "dino":
        model = DINOFERModel(model_variant="vits16", freeze=True)
    else:
        model = ConvNeXtFERModel(freeze_backbone=True)

    if checkpoint and os.path.exists(checkpoint):
        state = torch.load(checkpoint, map_location=device)
        model.load_state_dict(state)
        print(f"Loaded checkpoint: {checkpoint}")
    else:
        print("No checkpoint supplied – using randomly initialized head.")

    model.to(device)
    model.eval()
    return model


@torch.no_grad()
def predict_single(model, image_path: str, transform, device: torch.device):
    image = Image.open(image_path).convert("RGB")
    tensor = transform(image).unsqueeze(0).to(device)
    logits = model(tensor)
    probs  = torch.softmax(logits, dim=-1).squeeze(0)
    pred   = probs.argmax().item()
    return pred, probs.cpu().tolist()


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=["dino", "convnext"], default="dino")
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--image", default=None, help="Path to single image")
    p.add_argument("--image_dir", default=None, help="Path to folder of images")
    p.add_argument("--image_size", type=int, default=224)
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(args.model, args.checkpoint, device)
    transform = get_val_transforms(args.image_size)

    images_to_predict = []
    if args.image:
        images_to_predict = [args.image]
    elif args.image_dir:
        images_to_predict = [
            os.path.join(args.image_dir, f)
            for f in os.listdir(args.image_dir)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ]
    else:
        print("Provide --image or --image_dir")
        return

    print(f"\n{'='*55}")
    print(f"  Predictions using: {args.model.upper()}")
    print(f"{'='*55}\n")

    for img_path in images_to_predict:
        pred, probs = predict_single(model, img_path, transform, device)
        emo   = EMOTION_NAMES[pred]
        emoji = EMOTION_EMOJI[pred]
        conf  = probs[pred] * 100
        fname = os.path.basename(img_path)
        print(f"  {fname:<35} → {emoji} {emo:<10} ({conf:.1f}% confident)")
        # Top-3
        top3 = sorted(enumerate(probs), key=lambda x: -x[1])[:3]
        for idx, prob in top3:
            bar = "▓" * int(prob * 20)
            print(f"      {EMOTION_NAMES[idx]:<10} [{bar:<20}] {prob*100:.1f}%")
        print()


if __name__ == "__main__":
    main()
