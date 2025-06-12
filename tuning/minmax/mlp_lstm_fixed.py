import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import random
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import optuna
from sklearn.metrics import mean_squared_error
from models.lstm import LSTMRegressor
from utils.stock_data import get_data_loaders, StockIndex
from utils.early_stopping import EarlyStopping

SEED=42

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

class MLPEncoder(nn.Module):
    def __init__(self, input_dim, layer_dims, dropout=0.2, activation=nn.ReLU):
        super().__init__()
        layers = []
        in_dim = input_dim
        for i, out_dim in enumerate(layer_dims):
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


BEST_LSTM_PARAMS = {
    StockIndex.NVDA: {"hidden_dim": 352, "num_layers": 1, "dropout": 0.2854, "batch_size": 16, "lr": 0.000445},
    StockIndex.AAPL: {"hidden_dim": 384, "num_layers": 1, "dropout": 0.2291, "batch_size": 16, "lr": 0.002636},
    StockIndex.KO: {"hidden_dim": 480, "num_layers": 1, "dropout": 0.1947, "batch_size": 16, "lr": 0.001465},
}

def train_and_evaluate(trial, stock: StockIndex):
    lstm_params = BEST_LSTM_PARAMS[stock]
    lstm_hidden_dim = lstm_params["hidden_dim"]
    lstm_num_layers = lstm_params["num_layers"]
    lstm_dropout = lstm_params["dropout"]
    batch_size = lstm_params["batch_size"]
    lstm_lr = lstm_params["lr"]

    window_size = 7
    train_dl, val_dl, _, _ = get_data_loaders(stock, window_size, batch_size)
    input_dim = next(iter(train_dl))[0].shape[-1]

    # Build flexible MLP encoder config
    num_layers = trial.suggest_int("num_layers", 1, 3)
    layer_dims = []
    for i in range(num_layers):
        dim = trial.suggest_int(f"layer_{i}_dim", 16, 128,step=16)
        layer_dims.append(dim)
    
    # Ensure output dim matches LSTM input requirement
    layer_dims[-1] = input_dim
    dropout = trial.suggest_float("dropout", 0.1, 0.5)
    mlp_lr = trial.suggest_float("mlp_lr", 1e-4, 1e-2, log=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    mlp_encoder = MLPEncoder(input_dim=input_dim, layer_dims=layer_dims, dropout=dropout).to(device)
    lstm = LSTMRegressor(
        input_dim=input_dim,
        hidden_dim=lstm_hidden_dim,
        num_layers=lstm_num_layers,
        dropout=lstm_dropout
    ).to(device)

    optimizer = optim.Adam([
        {'params': mlp_encoder.parameters(), 'lr': mlp_lr},
        {'params': lstm.parameters(), 'lr': lstm_params['lr']}
    ])
    criterion = nn.MSELoss()
    early_stopper = EarlyStopping(patience=20, min_delta=1e-4)
    val_losses = []

    for epoch in range(50):
        mlp_encoder.train()
        lstm.train()
        for xb, yb in train_dl:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            encoded = mlp_encoder(xb)
            preds = lstm(encoded).squeeze()
            loss = criterion(preds, yb)
            loss.backward()
            optimizer.step()

        mlp_encoder.eval()
        lstm.eval()
        val_loss = 0.0
        with torch.no_grad():
            for xb, yb in val_dl:
                xb, yb = xb.to(device), yb.to(device)
                encoded = mlp_encoder(xb)
                preds = lstm(encoded).squeeze()
                val_loss += criterion(preds, yb).item()
        val_loss /= len(val_dl)
        val_losses.append(val_loss)

        early_stopper(val_loss)
        if early_stopper.early_stop:
            print(f"Stopped at epoch: {epoch}")
            break

    avg_val_loss = np.mean(val_losses[-5:])
    return avg_val_loss

def main():
    results_file = f"mlp_lstm_fixed_results.txt"

    with open(results_file, "w") as f:
        for stock in StockIndex:
            print(f"Tuning for {stock.name} (MLP encoder)...")
            f.write(f"\n===== {stock.name} (MLP encoder) =====\n")

            def objective(trial):
                try:
                    return train_and_evaluate(trial, stock)
                except Exception as e:
                    print(f"Trial failed for {stock.name}: {e}")
                    return float("inf")

            study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=SEED))
            study.optimize(objective, n_trials=150)

            print(f"Best trial for {stock.name}:")
            print(f"Value: {study.best_value:.6f}")
            print(f"Params: {study.best_trial.params}")
            f.write(f"Val Loss: {study.best_value:.6f}\n")
            for key, val in study.best_trial.params.items():
                print(f"{key}: {val}\n")
                f.write(f"{key}: {val}\n")

if __name__ == "__main__":
    main()
