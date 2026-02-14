import torch
import torch.nn as nn


class Attention(nn.Module):
    def __init__(self, feature_dim, hidden_dim, attention_dim):
        super().__init__()

        self.feature_att = nn.Linear(feature_dim, attention_dim)
        self.hidden_att = nn.Linear(hidden_dim, attention_dim)

        self.full_att = nn.Linear(attention_dim, 1)

        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, features, hidden_state):
        """
        features: (batch_size, 49, feature_dim)
        hidden_state: (batch_size, hidden_dim)
        """

        # Transform features
        att1 = self.feature_att(features)  # (batch_size, 49, attention_dim)

        # Transform hidden state
        att2 = self.hidden_att(hidden_state)  # (batch_size, attention_dim)

        # Expand hidden state to match spatial dimension
        att2 = att2.unsqueeze(1)  # (batch_size, 1, attention_dim)

        # Compute attention scores
        att = self.full_att(
            self.relu(att1 + att2)
        ).squeeze(2)  # (batch_size, 49)

        alpha = self.softmax(att)  # (batch_size, 49)

        # Compute weighted sum
        context = (features * alpha.unsqueeze(2)).sum(dim=1)

        return context, alpha

if __name__ == "__main__":
    att = Attention(2048, 512, 256)
    features = torch.randn(4, 49, 2048)
    hidden = torch.randn(4, 512)

    context, alpha = att(features, hidden)

    print(context.shape)  # (4, 2048)
    print(alpha.shape)    # (4, 49)
