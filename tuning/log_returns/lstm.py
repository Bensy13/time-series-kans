import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
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

def train_and_evaluate(trial, stock: StockIndex):
    window_size = 7
    hidden_dim = trial.suggest_int("hidden_dim", 32, 512, step=32)
    num_layers = trial.suggest_int("num_layers", 1, 3)
    dropout = trial.suggest_float("dropout", 0.1, 0.5)
    lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
    batch_size = trial.suggest_categorical("batch_size", [16, 32, 64])

    train_dl, val_dl, _ = get_log_return_loaders(stock, window_size, batch_size)
    input_dim = next(iter(train_dl))[0].shape[-1]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LSTMRegressor(input_dim=input_dim, hidden_dim=hidden_dim, num_layers=num_layers, dropout=dropout).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    criterion = nn.MSELoss()
    early_stopper = EarlyStopping(patience=20, min_delta=1e-5)

    val_losses = []

    for epoch in range(50):
        model.train()
        for xb, yb, _ in train_dl:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb).squeeze(), yb)
            loss.backward()
            optimizer.step()

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for xb, yb, _ in val_dl:
                xb, yb = xb.to(device), yb.to(device)
                val_loss += criterion(model(xb).squeeze(), yb).item()
        val_loss /= len(val_dl)
        val_losses.append(val_loss)

        scheduler.step(val_loss)
        early_stopper(val_loss)
        if early_stopper.early_stop:
            break

    avg_val_loss = np.mean(val_losses[-5:])  # average of last 5 validation losses
    return avg_val_loss

def main():
    set_seed(SEED)
    results_file = "log_return_lstm_best_results.txt"
    with open(results_file, "w") as f:
        for stock in StockIndex:
            print(f"\n--- Tuning LSTM on Log Returns for {stock.name} ---")

            def objective(trial):
                try:
                    return train_and_evaluate(trial, stock)
                except Exception as e:
                    print(f"Trial failed: {e}")
                    return float("inf")

            study = optuna.create_study(
                direction="minimize",
                sampler=optuna.samplers.TPESampler(seed=SEED)
            )
            study.optimize(objective, n_trials=150)

            print(f"Best trial for {stock.name}:")
            print(f"Validation Loss: {study.best_value:.6f}")
            print("Params:")
            for key, value in study.best_trial.params.items():
                print(f"  {key}: {value}")

            # Write to file
            f.write(f"Stock: {stock.name}\n")
            f.write(f"Val Loss: {study.best_value:.6f}\n")
            for key, value in study.best_trial.params.items():
                f.write(f"{key}: {value}\n")
            f.write("\n")

if __name__ == "__main__":
    main()
