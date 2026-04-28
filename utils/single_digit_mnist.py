import utils.compat
import os

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from typing import Optional


class SingleDigitMNISTDataset(Dataset):
    def __init__(
        self,
        csv_file: str,
        image_dir: Optional[str] = None,
        transform=None,
    ):
        self.image_dir = image_dir
        self.transform = transform
        self.df = pd.read_csv(csv_file)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        img_path = (
            os.path.join(self.image_dir, row["image_path"])
            if self.image_dir
            else row["image_path"]
        )
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        label = torch.tensor(row["label"], dtype=torch.long)
        visible = torch.tensor(row["visible"], dtype=torch.bool)
        return image, label, visible

    @property
    def data(self) -> np.ndarray:
        """Load all images into a (N, H, W, 3) uint8 numpy array."""
        images = []
        for _, row in self.df.iterrows():
            img_path = (
                os.path.join(self.image_dir, row["image_path"])
                if self.image_dir
                else row["image_path"]
            )
            img = np.array(Image.open(img_path).convert("RGB"), dtype=np.uint8)
            images.append(img)
        return np.stack(images)  # (N, H, W, 3) uint8

    @property
    def targets(self) -> np.ndarray:
        """Labels as a (N,) int64 numpy array."""
        return self.df["label"].values.astype(np.int64)
