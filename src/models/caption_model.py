import torch
import torch.nn as nn

from src.models.lstm_decoder import LSTMDecoder


class CaptionModel(nn.Module):
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

        self.decoder = LSTMDecoder(
            feature_dim=feature_dim,
            embed_dim=embed_dim,
            hidden_dim=hidden_dim,
            vocab_size=vocab_size,
            attention_dim=attention_dim,
            num_layers=num_layers,
            dropout=dropout,
        )

    def forward(self, features, captions):
        """
        features: (batch_size, 49, 2048)
        captions: (batch_size, max_len)
        """

        outputs = self.decoder(features, captions)

        return outputs

if __name__ == "__main__":
    vocab_size = 5000
    model = CaptionModel(2048, 256, 512, vocab_size, 256)

    features = torch.randn(4, 49, 2048)
    captions = torch.randint(0, vocab_size, (4, 30))

    outputs = model(features, captions)

    print(outputs.shape)
