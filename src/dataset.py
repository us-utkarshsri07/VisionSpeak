import os
import json
import pickle
import numpy as np
import torch
from torch.utils.data import Dataset

from src.config import PROCESSED_DATA_DIR, FEATURES_DIR


class CaptionDataset(Dataset):
    def __init__(self, split="train"):
        super().__init__()

        assert split in ["train", "val"], "Split must be 'train' or 'val'"

        self.split = split

        # Load train/val split
        split_path = os.path.join(PROCESSED_DATA_DIR, "train_val_split.json")
        with open(split_path, "r") as f:
            split_data = json.load(f)

        self.image_list = split_data[split]

        # Load encoded captions
        captions_path = os.path.join(PROCESSED_DATA_DIR, "encoded_captions.pkl")
        with open(captions_path, "rb") as f:
            self.encoded_captions = pickle.load(f)

        # Features directory
        self.features_dir = os.path.join(FEATURES_DIR, split)

        # Create (image_name, caption) pairs
        self.samples = self._build_samples()

    def _build_samples(self):
        samples = []

        for image_name in self.image_list:
            captions = self.encoded_captions[image_name]

            for caption in captions:
                samples.append((image_name, caption))

        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        image_name, caption = self.samples[idx]

        # Feature file should be image_name + ".npy"
        feature_file = image_name + ".npy"
        feature_path = os.path.join(self.features_dir, feature_file)

        if not os.path.exists(feature_path):
            raise FileNotFoundError(f"Feature file not found: {feature_path}")

        features = np.load(feature_path)

        features = torch.tensor(features, dtype=torch.float32)
        caption = torch.tensor(caption, dtype=torch.long)

        return features, caption

if __name__ == "__main__":
    dataset = CaptionDataset(split="train")
    print("Dataset size:", len(dataset))

    sample_features, sample_caption = dataset[0]

    print("Feature shape:", sample_features.shape)
    print("Caption shape:", sample_caption.shape)
