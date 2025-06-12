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
from models.kan_encoder import KANEncoder
from utils.stock_data import get_log_return_loaders, StockIndex
from utils.early_stopping import EarlyStopping

SEED = 42

def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

BEST_LSTM_PARAMS = {
    StockIndex.NVDA: {
        "hidden_dim": 192,
        "num_layers": 3,
        "dropout": 0.22832671613107025,
        "batch_size": 16,
        "lr": 0.001025807645705379
    },
    StockIndex.AAPL: {
        "hidden_dim": 32,
        "num_layers": 2,
        "dropout": 0.1879585812618099,
        "batch_size": 64,
        "lr": 0.00016388823896781193
    },
    StockIndex.KO: {
        "hidden_dim": 32,
        "num_layers": 1,
        "dropout": 0.15532586065977919,
        "batch_size": 64,
        "lr": 0.0010746245334922824
    }
}

def train_and_evaluate(trial, stock: StockIndex):
    lstm_params = BEST_LSTM_PARAMS[stock]
    window_size = 7
    batch_size = lstm_params["batch_size"]

    train_dl, val_dl, _ = get_log_return_loaders(stock, window_size, batch_size)
    input_dim = next(iter(train_dl))[0].shape[-1]

    # Fixed LSTM architecture
    lstm = LSTMRegressor(
        input_dim=input_dim,
        hidden_dim=lstm_params['hidden_dim'],
        num_layers=lstm_params['num_layers'],
        dropout=lstm_params['dropout']
    ).to(device := torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    # Tuneable KAN encoder
    kan_hidden_dim = trial.suggest_int("kan_hidden_dim", 4, 16)
    encoded_dim = input_dim
    grid = trial.suggest_int("grid", 7, 21, step=2)
    k = trial.suggest_categorical("k", [2, 3, 4, 5, 6])
    kan_lr = trial.suggest_float("kan_lr", 1e-4, 1e-2, log=True)
    lambda_reg = trial.suggest_float("lambda_reg", 1e-5, 5e-3, log=True)
    lamb_entropy = trial.suggest_float("lamb_entropy", 0.1, 2.0)
    lamb_l1 = trial.suggest_float("lamb_l1", 0.1, 2.0)

    kan_encoder = KANEncoder(
        width=[input_dim, kan_hidden_dim, encoded_dim],
        grid=grid,
        k=k,
        symbolic_enabled=False,
        affine_trainable=True,
        device=device,
        seed=SEED
    ).to(device)

    optimizer = optim.Adam([
        {"params": kan_encoder.parameters(), "lr": kan_lr},
        {"params": lstm.parameters(), "lr": lstm_params['lr']}
    ])
    criterion = nn.MSELoss()
    early_stopper = EarlyStopping(patience=20, min_delta=1e-5)
    val_losses = []

    for epoch in range(50):
        # --- Train ---
        kan_encoder.train()
        lstm.train()
        for xb, yb, _ in train_dl:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            encoded = kan_encoder(xb)
            preds = lstm(encoded).squeeze()
            loss = criterion(preds, yb)

            reg = kan_encoder.kan.get_reg(
                reg_metric="edge_forward_spline_n",
                lamb_l1=lamb_l1,
                lamb_entropy=lamb_entropy,
                lamb_coef=0.0,
                lamb_coefdiff=0.0
            )
            total_loss = loss + lambda_reg * reg
            total_loss.backward()
            optimizer.step()

        # --- Validate ---
        kan_encoder.eval()
        lstm.eval()
        val_loss = 0.0
        with torch.no_grad():
            for xb, yb, _ in val_dl:
                xb, yb = xb.to(device), yb.to(device)
                encoded = kan_encoder(xb)
                preds = lstm(encoded).squeeze()
                val_loss += criterion(preds, yb).item()
        val_loss /= len(val_dl)
        val_losses.append(val_loss)

        early_stopper(val_loss)
        if early_stopper.early_stop:
            print(f"Stopped early at epoch {epoch}")
            break

    return np.mean(val_losses[-5:])  # Smooth last 5 val losses

def main():
    set_seed(SEED)
    results_file = "kan_fixed_lstm_log_return_results.txt"

    with open(results_file, "w") as f:
        for stock in StockIndex:
            print(f"\n--- Tuning KAN encoder for {stock.name} ---")
            f.write(f"\n===== {stock.name} =====\n")

            def objective(trial):
                try:
                    return train_and_evaluate(trial, stock)
                except Exception as e:
                    print(f"Trial failed for {stock.name}: {e}")
                    return float("inf")

            study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=SEED))
            study.optimize(objective, n_trials=150)

            print(f"Best trial for {stock.name}:")
            print(f"Val Loss: {study.best_value:.6f}")
            print("Params:")
            for k, v in study.best_trial.params.items():
                print(f"{k}: {v}")
                f.write(f"{k}: {v}\n")
            f.write(f"Val Loss: {study.best_value:.6f}\n\n")

if __name__ == "__main__":
    main()
