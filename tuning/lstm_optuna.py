import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import torch.nn as nn
import torch.optim as optim
from utils.stock_data import get_data_loaders, StockIndex
from utils.early_stopping import EarlyStopping
from models.lstm import LSTMRegressor
import optuna

def train_and_evaluate(trial, stock: StockIndex):
    window_size = 7  # Fixed window size
    hidden_dim = trial.suggest_categorical("hidden_dim", [32, 64, 128, 256])
    num_layers = trial.suggest_int("num_layers", 1, 3)
    dropout = trial.suggest_float("dropout", 0.1, 0.5)
    lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
    batch_size = trial.suggest_categorical("batch_size", [16, 32, 64])

    train_dl, val_dl, test_dl, _ = get_data_loaders(stock, window_size, batch_size)
    input_dim = next(iter(train_dl))[0].shape[-1]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LSTMRegressor(input_dim=input_dim, hidden_dim=hidden_dim, num_layers=num_layers, dropout=dropout).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    early_stopper = EarlyStopping(patience=20, min_delta=1e-4)

    for epoch in range(100):
        model.train()
        for xb, yb in train_dl:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb).squeeze(), yb)
            loss.backward()
            optimizer.step()

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for xb, yb in val_dl:
                xb, yb = xb.to(device), yb.to(device)
                val_loss += criterion(model(xb).squeeze(), yb).item()
        val_loss /= len(val_dl)

        early_stopper(val_loss)
        if early_stopper.early_stop:
            break

    return val_loss

def main():
    stock = StockIndex.AAPL  # change for each run

    def objective(trial):
        return train_and_evaluate(trial, stock)

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=150)

    print("Best trial number:", study.best_trial.number)
    print("Best trial params:", study.best_trial.params)
    print("Best validation loss:", study.best_value)

if __name__ == "__main__":
    main()
