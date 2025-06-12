import torch.nn as nn

class MLPEncoder(nn.Module):
    def __init__(self, input_dim, layer_dims, dropout=0.2, activation=nn.ReLU):
        super().__init__()
        layers = []
        in_dim = input_dim
        for out_dim in layer_dims:
            layers.append(nn.Linear(in_dim, out_dim))
            layers.append(activation())
            layers.append(nn.Dropout(dropout))
            in_dim = out_dim
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        B, T, F = x.shape
        x_flat = x.view(B * T, F)
        out_flat = self.model(x_flat)
        return out_flat.view(B, T, -1)
