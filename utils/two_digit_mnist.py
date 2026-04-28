import os
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset


class TwoDigitMNISTDataset(Dataset):
    """
    Dataset for the two-digit MNIST using Pandas.
    """

    def __init__(
        self,
        csv_file: str,
        image_dir: str | None = None,
        image_col: str = "image_path",
        digit_prefix: str = "label",
        transform=None,
        use_masked_labels: bool = False,
        mask_symbol: float = -1.0,
    ):
        self.image_dir = image_dir
        self.transform = transform
        self.image_col = image_col
        self.digit_prefix = digit_prefix
        self.use_masked_labels = use_masked_labels
        self.mask_symbol = mask_symbol

        df = pd.read_csv(csv_file)

        clean_prefix = f"{digit_prefix}_"
        masked_prefix = f"masked_{digit_prefix}_"

        # Identify clean label columns
        clean_cols = sorted(
            [c for c in df.columns if c.startswith(clean_prefix)],
            key=lambda c: int(c.split("_")[-1]),
        )
        
        # Create a numpy array for clean labels and drop original columns from df for memory efficiency
        self._clean_labels = df[clean_cols].values.astype(np.float32)
        df = df.drop(columns=clean_cols)

        # Identify masked label columns
        masked_cols = sorted(
            [c for c in df.columns if c.startswith(masked_prefix)],
            key=lambda c: int(c.split("_")[-1]),
        )
        self._has_masked_labels = len(masked_cols) > 0

        if self._has_masked_labels:
            self._masked_labels = df[masked_cols].values.astype(np.float32)
            df = df.drop(columns=masked_cols)
        else:
            self._masked_labels = None

        self.labels_df = df

    @property
    def data(self) -> np.ndarray:
        """Load all images into a (N, H, W, 3) uint8 numpy array."""
        images = []
        for img_path in self.labels_df[self.image_col]:
            if self.image_dir:
                path = os.path.join(self.image_dir, img_path)
            else:
                path = img_path
            img = np.array(Image.open(path).convert("RGB"), dtype=np.uint8)
            images.append(img)
        return np.stack(images)

    @property
    def targets(self) -> np.ndarray:
        """Return multi-hot labels as a (N, n_class) float32 numpy array."""
        return self._clean_labels

    def __len__(self) -> int:
        return len(self.labels_df)

    def __getitem__(self, idx: int):
        row = self.labels_df.iloc[idx]

        img_path = row[self.image_col]
        if self.image_dir:
            img_path = os.path.join(self.image_dir, img_path)

        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)

        clean_label = torch.from_numpy(self._clean_labels[idx])

        if self.use_masked_labels and self._has_masked_labels:
            masked_label = torch.from_numpy(self._masked_labels[idx])
            # mask=1 where the value is observed, mask=0 where it was replaced
            mask = (masked_label != self.mask_symbol).float()
            return image, masked_label, mask

        # Default: clean label, all-ones mask
        return image, clean_label, torch.ones(clean_label.shape[0])
