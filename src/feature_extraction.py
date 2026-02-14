import os
import json
import torch
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
import numpy as np
from tqdm import tqdm

from src.config import RAW_DATA_DIR, PROCESSED_DATA_DIR, FEATURES_DIR


DEVICE = torch.device("cpu")


def get_encoder():
    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)

    # Remove final fully connected layer
    modules = list(model.children())[:-2]
    model = torch.nn.Sequential(*modules)

    model.eval()
    model.to(DEVICE)

    return model


def extract_features():
    # Load train-val split
    with open(os.path.join(PROCESSED_DATA_DIR, "train_val_split.json"), "r") as f:
        split_data = json.load(f)

    image_root = os.path.join(RAW_DATA_DIR, "images")

    encoder = get_encoder()

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    for split in ["train", "val"]:
        split_folder = os.path.join(FEATURES_DIR, split)
        os.makedirs(split_folder, exist_ok=True)

        for image_name in tqdm(split_data[split]):
            image_path = os.path.join(image_root, image_name)

            image = Image.open(image_path).convert("RGB")
            image = transform(image).unsqueeze(0).to(DEVICE)

            with torch.no_grad():
                features = encoder(image)

            # Shape: (1, 2048, 7, 7)
            features = features.squeeze(0)  # (2048, 7, 7)

            # Reshape to (49, 2048)
            features = features.permute(1, 2, 0).reshape(-1, 2048)

            feature_path = os.path.join(split_folder, image_name + ".npy")
            np.save(feature_path, features.cpu().numpy())

    print("Feature extraction complete.")


if __name__ == "__main__":
    extract_features()
