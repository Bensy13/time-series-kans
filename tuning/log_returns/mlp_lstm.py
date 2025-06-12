import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import random
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import optuna

from models.lstm import LSTMRegressor
from utils.stock_data import get_log_return_loaders, StockIndex
from utils.early_stopping import EarlyStopping

SEED = 42

def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

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

def train_mlp_lstm(trial, stock):
    window_size = 7
    batch_size = trial.suggest_categorical("batch_size", [16, 32, 64])
    train_dl, val_dl, _ = get_log_return_loaders(stock, window_size, batch_size)
    input_dim = next(iter(train_dl))[0].shape[-1]

    # MLP Encoder architecture
    num_layers = trial.suggest_int("num_layers", 1, 3)
    layer_dims = []
    for i in range(num_layers):
        dim = trial.suggest_int(f"layer_{i}_dim", 32, 256, step=32)
        layer_dims.append(dim)
    mlp_dropout = trial.suggest_float("mlp_dropout", 0.1, 0.5)
    mlp_lr = trial.suggest_float("mlp_lr", 1e-4, 1e-2, log=True)

    # LSTM architecture
    lstm_input_dim = layer_dims[-1]
    lstm_hidden_dim = trial.suggest_int("lstm_hidden_dim", 32, 512, step=32)
    lstm_num_layers = trial.suggest_int("lstm_num_layers", 1, 3)
    lstm_dropout = trial.suggest_float("lstm_dropout", 0.1, 0.5)
    lstm_lr = trial.suggest_float("lstm_lr", 1e-4, 1e-2, log=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    mlp_encoder = MLPEncoder(input_dim=input_dim, layer_dims=layer_dims, dropout=mlp_dropout).to(device)
    lstm = LSTMRegressor(
        input_dim=lstm_input_dim,
        hidden_dim=lstm_hidden_dim,
        num_layers=lstm_num_layers,
        dropout=lstm_dropout
    ).to(device)

    optimizer = optim.Adam([
        {'params': mlp_encoder.parameters(), 'lr': mlp_lr},
        {'params': lstm.parameters(), 'lr': lstm_lr}
    ])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    criterion = nn.MSELoss()
    early_stopper = EarlyStopping(patience=20, min_delta=1e-5)
    val_losses = []

    for epoch in range(50):
        mlp_encoder.train()
        lstm.train()
        for xb, yb, _ in train_dl:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            encoded = mlp_encoder(xb)
            preds = lstm(encoded).squeeze()
            loss = criterion(preds, yb)
            loss.backward()
            optimizer.step()

        # Validation
        mlp_encoder.eval()
        lstm.eval()
        val_loss = 0.0
        with torch.no_grad():
            for xb, yb, _ in val_dl:
                xb, yb = xb.to(device), yb.to(device)
                encoded = mlp_encoder(xb)
                preds = lstm(encoded).squeeze()
                val_loss += criterion(preds, yb).item()
        val_loss /= len(val_dl)
        val_losses.append(val_loss)

        scheduler.step(val_loss)
        early_stopper(val_loss)
        if early_stopper.early_stop:
            print(f"Early stopped at epoch {epoch}")
            break

    return np.mean(val_losses[-5:])

def main():
    set_seed(SEED)
    results_file = "mlp_lstm_log_return_results.txt"

    with open(results_file, "w") as f:
        for stock in StockIndex:
            print(f"\n--- Tuning MLP→LSTM on Log Returns for {stock.name} ---")
            f.write(f"\n===== {stock.name} (MLP→LSTM) =====\n")

            def objective(trial):
                try:
                    return train_mlp_lstm(trial, stock)
                except Exception as e:
                    print(f"Trial failed: {e}")
                    return float("inf")

            study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=SEED))
            study.optimize(objective, n_trials=150)

            print(f"Best trial for {stock.name}:")
            print(f"Validation Loss: {study.best_value:.6f}")
            print("Params:")
            for key, val in study.best_trial.params.items():
                print(f"{key}: {val}")
                f.write(f"{key}: {val}\n")
            f.write(f"Val Loss: {study.best_value:.6f}\n\n")

if __name__ == "__main__":
    main()
