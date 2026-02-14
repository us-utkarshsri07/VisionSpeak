import os
import json
import pickle
import numpy as np
import torch
from tqdm import tqdm
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

from src.config import *
from src.inference import load_model, load_vocab, generate_caption_beam


def load_ground_truth():
    with open(os.path.join(PROCESSED_DATA_DIR, "encoded_captions.pkl"), "rb") as f:
        encoded = pickle.load(f)

    with open(os.path.join(PROCESSED_DATA_DIR, "train_val_split.json"), "r") as f:
        split = json.load(f)

    return encoded, split["val"]


def decode_caption(encoded_caption, idx2word):
    words = []
    for idx in encoded_caption:
        word = idx2word.get(idx, None)   # <-- REMOVE str()
        if word in ["<start>", "<end>", "<pad>"] or word is None:
            continue
        words.append(word)
    return words



def evaluate():
    word2idx, idx2word = load_vocab()
    model = load_model(len(word2idx))

    encoded_data, val_images = load_ground_truth()

    bleu1_scores = []
    bleu4_scores = []
    lengths = []

    val_features_dir = os.path.join(FEATURES_DIR, "val")
    smoothie = SmoothingFunction().method4

    for image in tqdm(val_images[:200]):   # evaluate on first 200 for speed

        # base_name = os.path.splitext(image)[0]
        # feature_path = os.path.join(val_features_dir, base_name + ".npy")
        feature_path = os.path.join(val_features_dir, image + ".npy")

        if not os.path.exists(feature_path):
            continue

        features = np.load(feature_path)

        pred_caption, _ = generate_caption_beam(
            model,
            features,
            word2idx,
            idx2word,
            beam_size=3
        )

        pred_tokens = pred_caption.split()
        lengths.append(len(pred_tokens))

        references = []
        for gt in encoded_data[image]:
            references.append(decode_caption(gt, idx2word))

        print("Pred:", pred_tokens)
        print("Ref:", references[0])
        # break    
    
        
        bleu1 = sentence_bleu(
            references,
            pred_tokens,
            weights=(1, 0, 0, 0),
            smoothing_function=smoothie
        )

        bleu4 = sentence_bleu(
            references,
            pred_tokens,
            weights=(0.25, 0.25, 0.25, 0.25),
            smoothing_function=smoothie
        )

        bleu1_scores.append(bleu1)
        bleu4_scores.append(bleu4)

    print("\nEvaluation Results")
    print("BLEU-1:", np.mean(bleu1_scores))
    print("BLEU-4:", np.mean(bleu4_scores))
    print("Average Caption Length:", np.mean(lengths))


if __name__ == "__main__":
    evaluate()
