import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import random
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import optuna
from sklearn.metrics import mean_squared_error
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

# In global scope of your script
BEST_LSTM_PARAMS = {
    StockIndex.NVDA: {
        "hidden_dim": 352,
        "num_layers": 1,
        "dropout": 0.28541614143129757,
        "batch_size": 16,
        "lr": 0.00044517259602202056
    },
    StockIndex.AAPL: {
        "hidden_dim": 384,
        "num_layers": 1,
        "dropout": 0.22917171281782978,
        "batch_size": 16,
        "lr": 0.0026358901455780454
    },
    StockIndex.KO: {
        "hidden_dim": 480,
        "num_layers": 1,
        "dropout": 0.19466720673124002,
        "batch_size": 16,
        "lr": 0.0014651429936464527
    }
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

    kan_hidden_dim = trial.suggest_int("kan_hidden_dim", 4, 16)
    encoded_dim = input_dim # trial.suggest_categorical("encoded_dim", [4, 8, 16, 32]) -- keep LSTM structure the same
    grid = trial.suggest_int("grid", 7, 21, step=2)
    k = trial.suggest_categorical("k", [2, 3, 4, 5, 6])
    symbolic_enabled = False
    affine_trainable = True

    lambda_reg = trial.suggest_float("lambda_reg", 1e-5, 5e-3, log=True)
    lamb_entropy = trial.suggest_float("lamb_entropy", 0.1, 2.0)
    lamb_l1 = trial.suggest_float("lamb_l1", 0.1, 2.0)
    kan_lr = trial.suggest_float("kan_lr", 1e-4, 1e-2, log=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
        {'params': kan_encoder.parameters(), 'lr': kan_lr},
        {'params': lstm.parameters(), 'lr': lstm_lr}
    ])
    criterion = nn.MSELoss()
    early_stopper = EarlyStopping(patience=20, min_delta=1e-4)
    val_losses = []
    # --- Training loop ---
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

        # --- Validation evaluation ---
        kan_encoder.eval()
        lstm.eval()
        preds, targets = [], []
    
        val_loss = 0.0
        with torch.no_grad():
            for xb, yb in val_dl:
                xb, yb = xb.to(device), yb.to(device)
                encoded = kan_encoder(xb)
                preds = lstm(encoded).squeeze()
                val_loss += criterion(preds, yb).item()
        
        val_loss /= len(val_dl)
        val_losses.append(val_loss)

        early_stopper(val_loss)
        if early_stopper.early_stop:
            print(f"Stopped at epoch: {epoch}")
            break
    avg_val_loss = np.mean(val_losses[-5:])  # or last N epochs
    return avg_val_loss


def main():
    set_seed(SEED)
    results_file = f"kan_fixed_lstm_best_results_affine_true.txt"

    with open(results_file, "w") as f:
        for stock in StockIndex:
            print(f"Tuning for {stock.name}...")
            f.write(f"\n===== {stock.name} =====\n")

            def objective(trial):
                try:
                    return train_and_evaluate(trial, stock)
                except Exception as e:
                    print(f"Trial failed for {stock.name}: {e}")
                    return float("inf")

            study = optuna.create_study(
                direction="minimize",
                sampler=optuna.samplers.TPESampler(seed=SEED)
            )
            study.optimize(objective, n_trials=150)

            # Log best trial
            print(f"Best trial for {stock.name}:")
            print(f"Value: {study.best_value:.6f}")
            print(f"Params: {study.best_trial.params}")
            f.write(f"Val Loss: {study.best_value:.6f}\n")
            for key, val in study.best_trial.params.items():
                f.write(f"{key}: {val}\n")
if __name__ == "__main__":
    main()


