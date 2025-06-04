import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import random
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import optuna
from models.lstm import LSTMRegressor
from models.kan_encoder import KANEncoder
from utils.stock_data import get_data_loaders, StockIndex
from utils.early_stopping import EarlyStopping

SEED=42

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def train_kan_lstm(trial, stock):
    window_size = 7
    batch_size = trial.suggest_categorical("batch_size", [16, 32, 64])
    train_dl, val_dl, _, _ = get_data_loaders(stock, window_size, batch_size)
    input_dim = next(iter(train_dl))[0].shape[-1]

    # Jointly optimize KAN + LSTM architecture
    kan_hidden_dim = trial.suggest_int("kan_hidden_dim", 4, 16)
    encoded_dim = trial.suggest_categorical("encoded_dim", [4, 8, 16, 32])
    grid = trial.suggest_int("grid", 7, 21, step=2)
    k = trial.suggest_categorical("k", [2, 3, 4, 5, 6])
    kan_lr = trial.suggest_float("kan_lr", 1e-4, 1e-2, log=True)

    lstm_hidden_dim = trial.suggest_int("lstm_hidden_dim", 32, 512, step=32)
    lstm_num_layers = trial.suggest_int("lstm_num_layers", 1, 3)
    lstm_dropout = trial.suggest_float("lstm_dropout", 0.1, 0.5)
    lstm_lr = trial.suggest_float("lstm_lr", 1e-4, 1e-2, log=True)

    # Regularization and optimization
    lambda_reg = trial.suggest_float("lambda_reg", 1e-5, 5e-3, log=True)
    lamb_entropy = trial.suggest_float("lamb_entropy", 0.1, 2.0)
    lamb_l1 = trial.suggest_float("lamb_l1", 0.1, 2.0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    symbolic_enabled = False
    affine_trainable = False

    kan_encoder = KANEncoder(
        width=[input_dim, kan_hidden_dim, encoded_dim],
        grid=grid,
        symbolic_enabled=symbolic_enabled,
        device=device,
        affine_trainable=affine_trainable,
        k=k,
        seed=SEED
    ).to(device)

    lstm = LSTMRegressor(
        input_dim=encoded_dim,
        hidden_dim=lstm_hidden_dim,
        num_layers=lstm_num_layers,
        dropout=lstm_dropout
    ).to(device)

    optimizer = optim.Adam([
        {"params": kan_encoder.parameters(), "lr": kan_lr},
        {"params": lstm.parameters(), "lr": lstm_lr}
    ])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    criterion = nn.MSELoss()
    early_stopper = EarlyStopping(patience=20, min_delta=1e-4)
    val_losses = []

    for epoch in range(50):
        kan_encoder.train()
        lstm.train()
        for xb, yb in train_dl:
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

        # Validation
        kan_encoder.eval()
        lstm.eval()
        val_loss = 0.0
        with torch.no_grad():
            for xb, yb in val_dl:
                xb, yb = xb.to(device), yb.to(device)
                encoded = kan_encoder(xb)
                preds = lstm(encoded).squeeze()
                val_loss += criterion(preds, yb).item()
        val_loss /= len(val_dl)
        val_losses.append(val_loss)

        scheduler.step(val_loss)
        early_stopper(val_loss)
        if early_stopper.early_stop:
            break
    avg_val_loss = np.mean(val_losses[-5:])  # or last N epochs
    return avg_val_loss

def main():
    set_seed(SEED)
    results_file = f"kan_lstm_best_results_affine_false.txt"
    with open(results_file, "w") as f:
        for stock in StockIndex:
            print(f"Optimizing full KAN→LSTM architecture for {stock.name}...")
        
            def objective(trial):
                try:
                    return train_kan_lstm(trial, stock)
                except Exception as e:
                    print(f"Trial failed: {e}")
                    return float("inf")
        
            study = optuna.create_study(
                direction="minimize",
                sampler=optuna.samplers.TPESampler(seed=SEED))
            study.optimize(objective, n_trials=150)
        
            print(f"Best trial for {stock.name}:")
            print(f"Validation Loss: {study.best_value:.6f}")
            print(f"Params: {study.best_trial.params}")
            f.write(f"Val Loss: {study.best_value:.6f}\n")
            for key, value in study.best_trial.params.items():
                print(f"{key}: {value}")
                f.write(f"{key}: {value}\n")

if __name__ == "__main__":
    main()
