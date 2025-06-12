import os
import sys
import torch
import random
import numpy as np
import optuna
from optuna.trial import Trial
from torch import nn

# Add parent path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from utils.exchange_data import get_loaders

SEED = 42

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

class GRUModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        out, _ = self.gru(x)
        return self.fc(out[:, -1])

def train_and_evaluate(trial: Trial):
    set_seed(SEED)

    # Hyperparameter sampling
    window_size = 10
    horizon = 3
    hidden_dim = trial.suggest_int("hidden_dim", 32, 128, step=32)
    num_layers = trial.suggest_int("num_layers", 1, 3)
    lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
    batch_size = trial.suggest_int("batch_size", 32, 128, step=32)

    # Load data
    train_dl, val_dl, _, train_ds, val_ds, _ = get_loaders(
        window_size=window_size,
        horizon=horizon,
        batch_size=batch_size,
        standardize=True
    )

    input_dim = next(iter(train_dl))[0].shape[-1]
    output_dim = next(iter(train_dl))[1].shape[-1]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = GRUModel(input_dim, hidden_dim, num_layers, output_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    # Early stopping setup
    patience = 10
    min_delta = 1e-5
    best_val_loss = float("inf")
    bad_epochs = 0

    for epoch in range(100):
        model.train()
        for xb, yb in train_dl:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            optimizer.step()

        # Validation
        model.eval()
        total_loss = 0.0
        total_samples = 0
        preds, targets = [], []
        with torch.no_grad():
            for xb, yb in val_dl:
                xb, yb = xb.to(device), yb.to(device)
                y_pred = model(xb)
                total_loss += loss_fn(y_pred, yb).item() * xb.size(0)
                total_samples += xb.size(0)
                preds.append(y_pred.cpu())
                targets.append(yb)

        val_loss = total_loss / total_samples
        preds = torch.cat(preds, dim=0)
        targets = torch.cat(targets, dim=0)
        std_penalty = (preds.std() - targets.std()) ** 2
        full_loss = val_loss + 0.1 * std_penalty.item()

        if full_loss + min_delta < best_val_loss:
            best_val_loss = full_loss
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                break

    return best_val_loss

def main():
    print("\n--- Tuning GRU ---")

    def objective(trial):
        try:
            return train_and_evaluate(trial)
        except Exception as e:
            print(f"Trial failed: {e}")
            return float("inf")

    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=SEED))
    study.optimize(objective, n_trials=200)

    print("\n--- Best Trial ---")
    print(f"Trial #{study.best_trial.number}")
    print(f"Validation Loss: {study.best_value:.6f}")
    for key, value in study.best_trial.params.items():
        print(f"{key}: {value}")

if __name__ == "__main__":
    main()
