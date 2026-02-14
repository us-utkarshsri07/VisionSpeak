import torch
import torch.nn as nn
from src.models.attention import Attention


class LSTMDecoder(nn.Module):
    def __init__(
        self,
        feature_dim,
        embed_dim,
        hidden_dim,
        vocab_size,
        attention_dim,
        num_layers=1,
        dropout=0.3,
    ):
        super().__init__()

        self.attention = Attention(feature_dim, hidden_dim, attention_dim)

        self.embedding = nn.Embedding(vocab_size, embed_dim)

        self.lstm = nn.LSTMCell(embed_dim + feature_dim, hidden_dim)

        self.fc = nn.Linear(hidden_dim, vocab_size)

        self.dropout = nn.Dropout(dropout)

        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size

    # ---------- FIX: INSIDE CLASS ----------
    def init_hidden_state(self, features):
        batch_size = features.size(0)

        h = torch.zeros(batch_size, self.hidden_dim).to(features.device)
        c = torch.zeros(batch_size, self.hidden_dim).to(features.device)

        return h, c

    # ---------- FORWARD ----------
    def forward(self, features, captions):
        batch_size = features.size(0)
        max_len = captions.size(1)

        embeddings = self.embedding(captions)

        h = torch.zeros(batch_size, self.hidden_dim).to(features.device)
        c = torch.zeros(batch_size, self.hidden_dim).to(features.device)

        outputs = torch.zeros(batch_size, max_len, self.vocab_size).to(features.device)

        for t in range(max_len):

            context, alpha = self.attention(features, h)

            lstm_input = torch.cat([embeddings[:, t, :], context], dim=1)

            h, c = self.lstm(lstm_input, (h, c))

            out = self.fc(self.dropout(h))

            outputs[:, t, :] = out

        return outputs


if __name__ == "__main__":
    vocab_size = 5000
    decoder = LSTMDecoder(2048, 256, 512, vocab_size, 256)

    features = torch.randn(4, 49, 2048)
    captions = torch.randint(0, vocab_size, (4, 30))

    outputs = decoder(features, captions)

    print(outputs.shape)
