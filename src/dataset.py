import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2

from src.config import (
    IMAGE_SIZE, CLASS_NAMES,
    SEX_CATEGORIES, LOCALIZATION_CATEGORIES, NUM_TABULAR_FEATURES,
)


def get_train_transform():
    return A.Compose([
        A.Resize(256, 256),
        A.RandomCrop(IMAGE_SIZE, IMAGE_SIZE),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])


def get_val_transform():
    return A.Compose([
        A.Resize(IMAGE_SIZE, IMAGE_SIZE),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])


def encode_tabular(age: float, sex: str, localization: str) -> torch.Tensor:
    features = np.zeros(NUM_TABULAR_FEATURES, dtype=np.float32)

    # Age: normalize to [0, 1]
    features[0] = age / 100.0

    # Sex: one-hot encoding
    sex_lower = sex.lower() if isinstance(sex, str) else "unknown"
    sex_idx = SEX_CATEGORIES.index(sex_lower) if sex_lower in SEX_CATEGORIES else SEX_CATEGORIES.index("unknown")
    features[1 + sex_idx] = 1.0

    # Localization: one-hot encoding
    loc_lower = localization.lower() if isinstance(localization, str) else "unknown"
    loc_idx = (
        LOCALIZATION_CATEGORIES.index(loc_lower)
        if loc_lower in LOCALIZATION_CATEGORIES
        else LOCALIZATION_CATEGORIES.index("unknown")
    )
    features[1 + len(SEX_CATEGORIES) + loc_idx] = 1.0

    return torch.tensor(features, dtype=torch.float32)


class HAM10000Dataset(Dataset):
    def __init__(self, csv_path: str, transform=None):
        self.df = pd.read_csv(csv_path)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # Load image
        image = np.array(Image.open(row["image_path"]).convert("RGB"))
        if self.transform:
            image = self.transform(image=image)["image"]

        # Encode tabular features
        tabular = encode_tabular(row["age"], row["sex"], row["localization"])

        # Label
        label = int(row["label"])

        return image, tabular, label
