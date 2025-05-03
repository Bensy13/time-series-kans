import torch
import torch.nn as nn

class LSTMRegressor(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
        output_dim: int = 1
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )

        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # x: (batch, sequence_length, input_dim)
        _, (hn, _) = self.lstm(x)  # hn shape: (num_layers, batch, hidden_dim)
        out = self.fc(hn[-1])      # use last layer's hidden state
        return out.squeeze(-1)     # shape: (batch,) if output_dim = 1
