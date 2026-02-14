import os
import sys
import json
import torch
import numpy as np
import random
import torch.nn.functional as F

from src.config import *
from src.models.caption_model import CaptionModel



def load_vocab():
    vocab_path = os.path.join(PROCESSED_DATA_DIR, "vocabulary.json")
    with open(vocab_path, "r") as f:
        word2idx = json.load(f)

    idx2word = {idx: word for word, idx in word2idx.items()}

    return word2idx, idx2word


def load_model(vocab_size):
    model = CaptionModel(
        feature_dim=2048,
        embed_dim=EMBED_SIZE,
        hidden_dim=HIDDEN_SIZE,
        vocab_size=vocab_size,
        attention_dim=ATTENTION_DIM,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT,
    ).to(DEVICE)

    checkpoint_path = os.path.join(CHECKPOINT_DIR, "best_model.pth")
    model.load_state_dict(torch.load(checkpoint_path, map_location=DEVICE))

    model.eval()

    return model


# def generate_caption(model, features, word2idx, idx2word, max_len=30):
#     features = torch.tensor(features, dtype=torch.float32).unsqueeze(0).to(DEVICE)

#     caption_indices = [word2idx["<start>"]]

#     h = torch.zeros(1, HIDDEN_SIZE).to(DEVICE)
#     c = torch.zeros(1, HIDDEN_SIZE).to(DEVICE)

#     for _ in range(max_len):

#         current_word = torch.tensor([caption_indices[-1]]).to(DEVICE)

#         embedding = model.decoder.embedding(current_word)

#         context, _ = model.decoder.attention(features, h)

#         lstm_input = torch.cat([embedding.squeeze(1), context], dim=1)

#         h, c = model.decoder.lstm(lstm_input, (h, c))

#         output = model.decoder.fc(h)

#         predicted = output.argmax(dim=1).item()

#         if predicted == word2idx["<end>"]:
#             break

#         caption_indices.append(predicted)

#     caption_words = [idx2word[idx] for idx in caption_indices[1:]]

#     return " ".join(caption_words)


# def generate_caption(model, features, word2idx, idx2word, max_len=30):
#     features = torch.tensor(features, dtype=torch.float32).unsqueeze(0).to(DEVICE)

#     caption_indices = [word2idx["<start>"]]

#     h = torch.zeros(1, HIDDEN_SIZE).to(DEVICE)
#     c = torch.zeros(1, HIDDEN_SIZE).to(DEVICE)

#     for _ in range(max_len):

#         current_word = torch.tensor([caption_indices[-1]], dtype=torch.long).to(DEVICE)

#         embedding = model.decoder.embedding(current_word)  # (1, embed_dim)

#         context, _ = model.decoder.attention(features, h)

#         lstm_input = torch.cat([embedding, context], dim=1)

#         h, c = model.decoder.lstm(lstm_input, (h, c))

#         output = model.decoder.fc(h)

#         predicted = output.argmax(dim=1).item()

#         if predicted == word2idx["<end>"]:
#             break

#         caption_indices.append(predicted)

#     caption_words = []
#     for idx in caption_indices[1:]:
#         word = idx2word[idx]
#         if word not in ["<start>", "<pad>"]:
#             caption_words.append(word)

#     return " ".join(caption_words)




# def generate_caption_beam(model, features, word2idx, idx2word, beam_size=3, max_len=30):
#     features = torch.tensor(features, dtype=torch.float32).unsqueeze(0).to(DEVICE)

#     start_token = word2idx["<start>"]
#     end_token = word2idx["<end>"]

#     beams = [( [start_token], 0.0,
#                torch.zeros(1, HIDDEN_SIZE).to(DEVICE),
#                torch.zeros(1, HIDDEN_SIZE).to(DEVICE) )]

#     completed = []

#     for _ in range(max_len):
#         new_beams = []

#         for seq, score, h, c in beams:

#             if seq[-1] == end_token:
#                 completed.append((seq, score))
#                 continue

#             current_word = torch.tensor([seq[-1]]).to(DEVICE)

#             embedding = model.decoder.embedding(current_word)

#             context, _ = model.decoder.attention(features, h)

#             lstm_input = torch.cat([embedding, context], dim=1)

#             h_new, c_new = model.decoder.lstm(lstm_input, (h, c))

#             output = model.decoder.fc(h_new)

#             log_probs = F.log_softmax(output, dim=1)

#             top_log_probs, top_indices = torch.topk(log_probs, beam_size)

#             for i in range(beam_size):
#                 next_word = top_indices[0][i].item()
#                 new_score = score + top_log_probs[0][i].item()

#                 new_seq = seq + [next_word]

#                 new_beams.append((new_seq, new_score, h_new, c_new))

#         beams = sorted(new_beams, key=lambda x: x[1], reverse=True)[:beam_size]

#     if completed:
#         best_seq = sorted(completed, key=lambda x: x[1], reverse=True)[0][0]
#     else:
#         best_seq = beams[0][0]

#     caption_words = []
#     for idx in best_seq[1:]:
#         word = idx2word[idx]
#         if word not in ["<start>", "<end>", "<pad>"]:
#             caption_words.append(word)

#     return " ".join(caption_words)

# def generate_caption_beam(model, features, word2idx, idx2word, beam_size=3, max_len=30):
    
#     with torch.no_grad():
#         features = torch.tensor(features, dtype=torch.float32).unsqueeze(0).to(DEVICE)

#     start_token = word2idx["<start>"]
#     end_token = word2idx["<end>"]

#     beams = [( [start_token], 0.0,
#                torch.zeros(1, HIDDEN_SIZE).to(DEVICE),
#                torch.zeros(1, HIDDEN_SIZE).to(DEVICE),
#                [] )]

#     completed = []

#     for _ in range(max_len):
#         new_beams = []

#         for seq, score, h, c, attn_list in beams:

#             if seq[-1] == end_token:
#                 completed.append((seq, score, attn_list))
#                 continue

#             current_word = torch.tensor([seq[-1]]).to(DEVICE)

#             embedding = model.decoder.embedding(current_word)

#             context, alpha = model.decoder.attention(features, h)

#             lstm_input = torch.cat([embedding, context], dim=1)

#             h_new, c_new = model.decoder.lstm(lstm_input, (h, c))

#             output = model.decoder.fc(h_new)

#             log_probs = torch.log_softmax(output, dim=1)

#             top_log_probs, top_indices = torch.topk(log_probs, beam_size)

#             for i in range(beam_size):
#                 next_word = top_indices[0][i].item()
#                 new_score = score + top_log_probs[0][i].item()

#                 new_seq = seq + [next_word]
#                 new_attn = attn_list + [alpha.squeeze(0).detach().cpu().numpy()]

#                 new_beams.append((new_seq, new_score, h_new, c_new, new_attn))

#         beams = sorted(new_beams, key=lambda x: x[1], reverse=True)[:beam_size]

#     if completed:
#         best_seq, _, best_attn = sorted(completed, key=lambda x: x[1], reverse=True)[0]
#     else:
#         best_seq, _, _, _, best_attn = beams[0]

#     caption_words = []
#     for idx in best_seq[1:]:
#         word = idx2word[idx]
#         if word not in ["<start>", "<end>", "<pad>"]:
#             caption_words.append(word)

#     return " ".join(caption_words), best_attn

import torch
import torch.nn.functional as F


# def generate_caption_beam(
#     model,
#     features,
#     word2idx,
#     idx2word,
#     beam_size=3,
#     max_len=20
# ):

#     model.eval()

#     device = next(model.parameters()).device

#     # Prepare feature tensor
#     features = torch.tensor(features, dtype=torch.float32).unsqueeze(0).to(device)

#     start_token = word2idx["<start>"]
#     end_token = word2idx["<end>"]

#     with torch.no_grad():

#         # Initialize hidden state
#         h, c = model.decoder.init_hidden_state(features)

#         # Initial beam
#         beams = [([start_token], 0.0, h, c, [])]

#         for _ in range(max_len):

#             new_beams = []

#             for seq, score, h_prev, c_prev, attn_list in beams:

#                 if seq[-1] == end_token:
#                     new_beams.append((seq, score, h_prev, c_prev, attn_list))
#                     continue

#                 last_word = torch.tensor([seq[-1]]).to(device)

#                 embeddings = model.decoder.embedding(last_word)

#                 context, alpha = model.decoder.attention(features, h_prev)

#                 lstm_input = torch.cat((embeddings, context), dim=1)

#                 h_new, c_new = model.decoder.lstm(lstm_input, (h_prev, c_prev))

#                 output = model.decoder.fc(h_new)

#                 log_probs = F.log_softmax(output, dim=1)

#                 unk_idx = word2idx["<unk>"]
#                 log_probs[:, unk_idx] = -1e9


#                 top_log_probs, top_indices = log_probs.topk(beam_size)

#                 # for i in range(beam_size):

#                 #     next_word = top_indices[0][i].item()
#                 #     new_score = score + top_log_probs[0][i].item()

#                 #     new_seq = seq + [next_word]

#                 # for i in range(beam_size):
#                 #     next_word = top_indices[0][i].item()
#                 #     new_score = score + top_log_probs[0][i].item()

#                 #     new_seq = seq + [next_word]
#                 for i in range(beam_size):
#                      next_word = top_indices[0][i].item()
#                      new_score = score + top_log_probs[0][i].item()
#                      # Prevent immediate repetition (strong rule)
#                      if len(seq) > 0 and next_word == seq[-1]:
#                           continue
                     
                     
#                      if next_word in seq:
#         new_score -= 0.5

#     new_seq = seq + [next_word]


#                     # Detach attention properly
#                     new_attn = attn_list + [
#                         alpha.squeeze(0).detach().cpu().numpy()
#                     ]

#                     new_beams.append((new_seq, new_score, h_new, c_new, new_attn))

#             beams = sorted(new_beams, key=lambda x: x[1], reverse=True)[:beam_size]

#         best_seq, _, _, _, best_attn = beams[0]

#         caption_words = []
#         for idx in best_seq:
#             word = idx2word[idx]
#             if word not in ["<start>", "<end>", "<pad>"]:
#                 caption_words.append(word)

#         caption = " ".join(caption_words)
#         best_attn = best_attn[:len(caption_words)]

#         return caption, best_attn

def generate_caption_beam(
    model,
    features,
    word2idx,
    idx2word,
    beam_size=5,
    max_len=20
):

    model.eval()
    device = next(model.parameters()).device

    # Prepare feature tensor
    features = torch.tensor(features, dtype=torch.float32).unsqueeze(0).to(device)

    start_token = word2idx["<start>"]
    end_token = word2idx["<end>"]
    unk_idx = word2idx["<unk>"]

    with torch.no_grad():

        # Initialize hidden state
        h, c = model.decoder.init_hidden_state(features)

        # Initial beam: (sequence, score, h, c, attention_list)
        beams = [([start_token], 0.0, h, c, [])]
        # if len(beams) == 0:
        #     break

        for _ in range(max_len):

            new_beams = []

            for seq, score, h_prev, c_prev, attn_list in beams:

                # If already ended, keep as is
                if seq[-1] == end_token:
                    new_beams.append((seq, score, h_prev, c_prev, attn_list))
                    continue

                last_word = torch.tensor([seq[-1]]).to(device)
                embedding = model.decoder.embedding(last_word)

                context, alpha = model.decoder.attention(features, h_prev)

                lstm_input = torch.cat((embedding, context), dim=1)
                h_new, c_new = model.decoder.lstm(lstm_input, (h_prev, c_prev))

                output = model.decoder.fc(h_new)
                log_probs = F.log_softmax(output, dim=1)

                # Block <unk>
                log_probs[:, unk_idx] = -1e9

                top_log_probs, top_indices = log_probs.topk(beam_size)

                for i in range(beam_size):

                    next_word = top_indices[0][i].item()
                    new_score = score + top_log_probs[0][i].item()

                    # # Prevent immediate repetition
                    # if len(seq) > 0 and next_word == seq[-1]:
                    #     continue

                    # Penalize repeated words (soft constraint)
                    if next_word in seq:
                        new_score -= 0.7

                    new_seq = seq + [next_word]

                    new_attn = attn_list + [
                        alpha.squeeze(0).detach().cpu().numpy()
                    ]

                    new_beams.append((new_seq, new_score, h_new, c_new, new_attn))

            # # Keep top beams
            # beams = sorted(new_beams, key=lambda x: x[1], reverse=True)[:beam_size]
            # beams = sorted(new_beams, key=lambda x: x[1] / len(x[0]), reverse=True)
            beams = sorted(new_beams, key=lambda x: x[1], reverse=True)[:beam_size]

        best_seq, _, _, _, best_attn = beams[0]

        # Convert indices to words
        caption_words = []
        for idx in best_seq:
            word = idx2word[idx]
            if word not in ["<start>", "<end>", "<pad>"]:
                caption_words.append(word)

        caption = " ".join(caption_words)

        # Align attention length
        best_attn = best_attn[:len(caption_words)]

        return caption, best_attn


def main():
    print("STARTING INFERENCE")
    word2idx, idx2word = load_vocab()
    vocab_size = len(word2idx)

    model = load_model(vocab_size)

    # Pick one validation image
    val_features_dir = os.path.join(FEATURES_DIR, "val")
    image_file = random.choice(os.listdir(val_features_dir))

    feature_path = os.path.join(val_features_dir, image_file)
    features = np.load(feature_path)

    caption,_ = generate_caption_beam(model, features, word2idx, idx2word)

    print("Generated Caption:",caption)
    # print(caption)


if __name__ == "__main__":
    main()
