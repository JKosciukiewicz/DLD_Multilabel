import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class BBBC021Dataset(Dataset):
    def __init__(
        self,
        data_file: str,
        split: str,
        transform=None,
        mask_uncertain: bool = True,
    ):
        """
        BBBC021 dataset for predicting Mechanism of Action (MoA) from morphological features.
        Uses Pandas for data loading.

        Args:
            data_file (str): Path to CSV with morphology features
            split (str): 'train' or 'test'
            transform (callable, optional): Optional transform to apply to features
            mask_uncertain (bool): If True, mask uncertain (0) labels in the target
        """
        self.transform = transform
        self.mask_uncertain = mask_uncertain

        print(
            f"Loading BBBC021 dataset using Pandas for split='{split}'..."
        )

        # Read CSV file
        data_df = pd.read_csv(data_file)

        # Identify feature and MoA columns
        self.feature_columns = [
            col for col in data_df.columns if col.startswith("feat_")
        ]
        self.moa_columns = [col for col in data_df.columns if col.startswith("moa_")]

        print(
            f"Found {len(self.feature_columns)} feature columns and {len(self.moa_columns)} MoA columns"
        )

        # Filter by split and drop nulls
        data_df = data_df[data_df["split"] == split].dropna()

        print(f"Loaded {len(data_df)} samples for split='{split}'")

        # Convert to numpy arrays for efficient storage
        print("Converting to numpy arrays...")
        self.features_data = (
            data_df[self.feature_columns].values.astype(np.float32)
        )
        raw_targets = data_df[self.moa_columns].values.astype(np.float32)

        # Create masks for uncertain labels if needed
        if self.mask_uncertain:
            self.masks_data = (raw_targets != 0).astype(np.float32)
        else:
            self.masks_data = np.ones_like(raw_targets, dtype=np.float32)

        # Convert -1 to 0 for binary classification
        self.targets_data = np.where(raw_targets < 0, 0, raw_targets)

        print(f"Feature dimension: {len(self.feature_columns)}")
        print(f"Target dimension: {len(self.moa_columns)}")

    def __len__(self):
        return len(self.features_data)

    def __getitem__(self, idx):
        features = torch.tensor(self.features_data[idx], dtype=torch.float32)
        targets = torch.tensor(self.targets_data[idx], dtype=torch.float32)
        mask = torch.tensor(self.masks_data[idx], dtype=torch.float32)

        if self.transform:
            features = self.transform(features)

        return features, targets, mask

    def get_feature_names(self):
        return self.feature_columns

    def get_target_names(self):
        return self.moa_columns


if __name__ == "__main__":
    # Test both train and test splits
    train_dataset = BBBC021Dataset(
        data_file="_data/BBBC021/BBBC021_dataset_complete_one_fold.csv", split="train"
    )
    test_dataset = BBBC021Dataset(
        data_file="_data/BBBC021/BBBC021_dataset_complete_one_fold.csv", split="test"
    )

    print(f"Training set: {len(train_dataset)} samples")
    print(f"Test set: {len(test_dataset)} samples")
    print(f"Feature names: {len(train_dataset.get_feature_names())} features")
    print(f"Target names: {len(train_dataset.get_target_names())} targets")
