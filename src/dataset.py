"""
dataset.py
----------
PyTorch Dataset classes for the Facial Emotion Recognition dataset.

Dataset structure expected:
    DATASET/
        train/
            1/         <- training images for class 1 (Surprise)
            2/         <- Fear
            ...
            7/         <- Neutral
        test/
            1/         <- test images for class 1
            2/
            ...
            7/
    train_labels.csv   <- columns: image, label  (labels 1-7)
    test_labels.csv    <- columns: image, label  (labels 1-7)

Emotion label mapping (RAF-DB style, 1-indexed):
    1 -> Surprise
    2 -> Fear
    3 -> Disgust
    4 -> Happy
    5 -> Sad
    6 -> Anger
    7 -> Neutral
"""

import os
import csv
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms

# ── label mapping ────────────────────────────────────────────────────────────
# RAF-DB uses 1-indexed labels; we convert to 0-indexed for CrossEntropy
LABEL_MAP = {
    1: "Surprise",
    2: "Fear",
    3: "Disgust",
    4: "Happy",
    5: "Sad",
    6: "Anger",
    7: "Neutral",
}
NUM_CLASSES = len(LABEL_MAP)


# ── augmentation helpers ──────────────────────────────────────────────────────
def get_train_transforms(image_size: int = 224):
    """Strong augmentation for training to reduce overfitting."""
    return transforms.Compose([
        transforms.Resize((image_size + 32, image_size + 32)),
        transforms.RandomCrop(image_size),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
        transforms.RandomRotation(15),
        transforms.RandomGrayscale(p=0.05),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])


def get_val_transforms(image_size: int = 224):
    """Minimal transforms for validation / test."""
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])


# ── Dataset: CSV-based (works for both train and test) ────────────────────────
class FERTrainDataset(Dataset):
    """
    Reads image filenames and integer labels from a CSV file.
    Labels in the CSV are 1-indexed; we shift them to 0-indexed internally.

    Images are stored in class sub-folders:
        img_dir/{label_1indexed}/filename.jpg
    """

    def __init__(self, csv_path: str, img_dir: str, transform=None):
        self.img_dir = img_dir
        self.transform = transform
        self.samples = []

        with open(csv_path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                fname = row["image"].strip()
                label_1idx = int(row["label"].strip())
                label_0idx = label_1idx - 1  # 0-indexed for CrossEntropy
                # Images live in sub-folder named by the 1-indexed label
                img_path = os.path.join(img_dir, str(label_1idx), fname)
                self.samples.append((img_path, label_0idx))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label


# ── Dataset: Test (folder-based) ─────────────────────────────────────────────
class FERTestDataset(Dataset):
    """
    Reads images from sub-folders named 1..7 inside test_dir.
    Folder names are 1-indexed; we shift to 0-indexed.
    """

    def __init__(self, test_dir: str, transform=None):
        self.transform = transform
        self.samples = []

        for class_folder in sorted(os.listdir(test_dir)):
            class_path = os.path.join(test_dir, class_folder)
            if not os.path.isdir(class_path):
                continue
            label = int(class_folder) - 1  # 0-indexed
            for fname in os.listdir(class_path):
                if fname.lower().endswith((".jpg", ".jpeg", ".png")):
                    self.samples.append((os.path.join(class_path, fname), label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label


# ── DataLoader factory ────────────────────────────────────────────────────────
def build_dataloaders(
    csv_path: str,
    train_img_dir: str,
    test_dir: str,
    batch_size: int = 32,
    num_workers: int = 0,
    image_size: int = 224,
    val_split: float = 0.1,
    seed: int = 42,
):
    """
    Returns train_loader, val_loader, test_loader.
    Splits the training CSV into train/val using val_split fraction.
    """
    # Two separate dataset objects so train gets augmentation and val does not.
    # Sharing a single object and swapping .transform mid-split is a bug
    # (random_split keeps a reference to the same Dataset, so the swap affects
    # both halves).
    train_data = FERTrainDataset(csv_path, train_img_dir,
                                 transform=get_train_transforms(image_size))
    val_data   = FERTrainDataset(csv_path, train_img_dir,
                                 transform=get_val_transforms(image_size))

    n         = len(train_data)
    val_size  = int(n * val_split)
    train_size = n - val_size

    generator = torch.Generator().manual_seed(seed)
    indices   = torch.randperm(n, generator=generator).tolist()

    train_ds = Subset(train_data, indices[val_size:])
    val_ds   = Subset(val_data,   indices[:val_size])

    test_ds = FERTestDataset(test_dir, transform=get_val_transforms(image_size))

    train_loader = DataLoader(train_ds, batch_size=batch_size,
                              shuffle=True, num_workers=num_workers,
                              pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size,
                              shuffle=False, num_workers=num_workers,
                              pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size,
                              shuffle=False, num_workers=num_workers,
                              pin_memory=True)

    print(f"  Train samples : {train_size}")
    print(f"  Val   samples : {val_size}")
    print(f"  Test  samples : {len(test_ds)}")
    return train_loader, val_loader, test_loader
