import os
import json
import pickle
import re
from collections import Counter
from tqdm import tqdm

from src.config import RAW_DATA_DIR, PROCESSED_DATA_DIR, MAX_CAPTION_LENGTH, FREQ_THRESHOLD, TRAIN_SPLIT


SPECIAL_TOKENS = {
    "<pad>": 0,
    "<start>": 1,
    "<end>": 2,
    "<unk>": 3,
}


def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-zA-Z ]+", "", text)
    return text.strip()


def load_captions():
    captions_file = os.path.join(RAW_DATA_DIR, "captions.txt")

    image_captions = {}

    with open(captions_file, "r") as f:
        next(f)  # skip header line
        for line in f:
            image_name, caption = line.strip().split(",", 1)
            caption = clean_text(caption)

            if image_name not in image_captions:
                image_captions[image_name] = []

            image_captions[image_name].append(caption)

    return image_captions


def build_vocab(image_captions):
    counter = Counter()

    for captions in image_captions.values():
        for caption in captions:
            counter.update(caption.split())

    vocab = dict(SPECIAL_TOKENS)

    idx = len(SPECIAL_TOKENS)

    for word, freq in counter.items():
        if freq >= FREQ_THRESHOLD:
            vocab[word] = idx
            idx += 1

    return vocab


def encode_caption(caption, vocab):
    tokens = caption.split()

    encoded = [vocab["<start>"]]

    for token in tokens:
        encoded.append(vocab.get(token, vocab["<unk>"]))

    encoded.append(vocab["<end>"])

    if len(encoded) < MAX_CAPTION_LENGTH:
        encoded += [vocab["<pad>"]] * (MAX_CAPTION_LENGTH - len(encoded))
    else:
        encoded = encoded[:MAX_CAPTION_LENGTH]

    return encoded


def preprocess():
    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)

    image_captions = load_captions()

    vocab = build_vocab(image_captions)

    encoded_data = {}

    for image, captions in tqdm(image_captions.items()):
        encoded_data[image] = [
            encode_caption(caption, vocab) for caption in captions
        ]

    # train-val split
    images = list(encoded_data.keys())
    split_index = int(len(images) * TRAIN_SPLIT)

    train_images = images[:split_index]
    val_images = images[split_index:]

    split_data = {
        "train": train_images,
        "val": val_images,
    }

    with open(os.path.join(PROCESSED_DATA_DIR, "vocabulary.json"), "w") as f:
        json.dump(vocab, f)

    with open(os.path.join(PROCESSED_DATA_DIR, "encoded_captions.pkl"), "wb") as f:
        pickle.dump(encoded_data, f)

    with open(os.path.join(PROCESSED_DATA_DIR, "train_val_split.json"), "w") as f:
        json.dump(split_data, f)

    print("Preprocessing complete.")
    print("Vocab size:", len(vocab))



if __name__ == "__main__":
    preprocess()
