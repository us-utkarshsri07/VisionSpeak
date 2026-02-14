import os
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.config import *
from src.dataset import CaptionDataset
from src.models.caption_model import CaptionModel


def load_vocab_size():
    vocab_path = os.path.join(PROCESSED_DATA_DIR, "vocabulary.json")
    with open(vocab_path, "r") as f:
        vocab = json.load(f)
    return len(vocab)


def train_one_epoch(model, loader, criterion, optimizer):
    model.train()
    total_loss = 0

    for features, captions in tqdm(loader):
        features = features.to(DEVICE)
        captions = captions.to(DEVICE)

        optimizer.zero_grad()

        # outputs = model(features, captions)

        # # Flatten for loss
        # outputs = outputs.view(-1, outputs.size(-1))
        # captions = captions.view(-1)

        # loss = criterion(outputs, captions)
        outputs = model(features, captions[:, :-1])
        targets = captions[:, 1:]
        outputs = outputs.reshape(-1, outputs.size(-1))
        targets = targets.reshape(-1)
        loss = criterion(outputs, targets)


        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


# def validate(model, loader, criterion):
#     model.eval()
#     total_loss = 0

#     with torch.no_grad():
#         for features, captions in loader:
#             features = features.to(DEVICE)
#             captions = captions.to(DEVICE)

#             outputs = model(features, captions)

#             outputs = outputs.view(-1, outputs.size(-1))
#             captions = captions.view(-1)

#             loss = criterion(outputs, captions)

#             total_loss += loss.item()

#     return total_loss / len(loader)
def validate(model, loader, criterion):
    model.eval()
    total_loss = 0

    with torch.no_grad():
        for features, captions in loader:
            features = features.to(DEVICE)
            captions = captions.to(DEVICE)

            outputs = model(features, captions[:, :-1])
            targets = captions[:, 1:]

            outputs = outputs.reshape(-1, outputs.size(-1))
            targets = targets.reshape(-1)

            loss = criterion(outputs, targets)

            total_loss += loss.item()

    return total_loss / len(loader)



def main():
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    vocab_size = load_vocab_size()

    train_dataset = CaptionDataset(split="train")
    train_dataset.samples = train_dataset.samples[:5000]
    val_dataset = CaptionDataset(split="val")
    val_dataset.samples = val_dataset.samples[:1000]

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    model = CaptionModel(
        feature_dim=2048,
        embed_dim=EMBED_SIZE,
        hidden_dim=HIDDEN_SIZE,
        vocab_size=vocab_size,
        attention_dim=ATTENTION_DIM,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT,
    ).to(DEVICE)

    criterion = nn.CrossEntropyLoss(ignore_index=0)  # ignore <pad>

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    best_val_loss = float("inf")

    for epoch in range(NUM_EPOCHS):
        print(f"\nEpoch {epoch+1}/{NUM_EPOCHS}")

        train_loss = train_one_epoch(model, train_loader, criterion, optimizer)
        val_loss = validate(model, val_loader, criterion)

        print(f"Train Loss: {train_loss:.4f}")
        print(f"Val Loss: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(
                model.state_dict(),
                os.path.join(CHECKPOINT_DIR, "best_model.pth"),
            )
            print("Best model saved.")

    print("Training complete.")


if __name__ == "__main__":
    main()
