import os
import sys
import torch
import random
import numpy as np
import optuna
from optuna.trial import Trial

# Add parent path to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from utils.exchange_data import get_loaders
from models.kangru import KANGRU

SEED = 42

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def train_and_evaluate(trial: Trial):
    set_seed(SEED)

    # Hyperparams to tune
    window_size = 10
    horizon = 3
    hidden_dim = trial.suggest_int("hidden_dim", 32, 128, step=32)
    num_layers = trial.suggest_int("num_layers", 1, 3)
    grid = trial.suggest_int("grid", 5, 17)
    k = trial.suggest_int("k", 2, 5)
    lr = trial.suggest_float("lr", 1e-4, 1e-1, log=True)
    lamb = trial.suggest_float("lambda_reg", 1e-5, 1e-2, log=True)
    lamb_entropy = trial.suggest_float("lamb_entropy", 0.1, 2.0)
    lamb_l1 = trial.suggest_float("lamb_l1", 1e-2, 2.0)
    lamb_coef = trial.suggest_float("lamb_coef", 1e-5, 1e-2, log=True)
    lamb_coefdiff = trial.suggest_float("lamb_coefdiff", 1e-5, 1e-2, log=True)
    batch_size = trial.suggest_int("batch_size", 32, 128, step=32)

    # Load data
    train_dl, val_dl, _, train_ds, val_ds, _ = get_loaders(
        window_size=window_size,
        horizon=horizon,
        batch_size=batch_size
    )

    input_dim = next(iter(train_dl))[0].shape[-1]
    output_dim = next(iter(train_dl))[1].shape[-1]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = KANGRU(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        output_dim=output_dim,
        grid=grid,
        k=k,
        device=device
    )

    model.fit(
        train_dl=train_dl,
        val_dl=val_dl,
        steps=30,
        lr=lr,
        lamb=lamb,
        lamb_l1=lamb_l1,
        lamb_entropy=lamb_entropy,
        lamb_coef=lamb_coef,
        lamb_coefdiff=lamb_coefdiff,
        patience=5,
        min_delta=1e-6,
        log=10,
        warmup_steps=3
    )

    # Validation error + variance penalty
    model.eval()
    preds, targets = [], []
    total_loss, total_samples = 0.0, 0
    loss_fn = torch.nn.MSELoss(reduction='none')

    with torch.no_grad():
        for xb, yb in val_dl:
            xb, yb = xb.to(device), yb.to(device)
            y_pred = model(xb)
            loss = loss_fn(y_pred, yb)
            total_loss += loss.sum().item()
            total_samples += yb.numel()
            preds.append(y_pred.cpu())
            targets.append(yb)

    preds = torch.cat(preds, dim=0)
    targets = torch.cat(targets, dim=0)

    avg_val_loss = total_loss / total_samples
    std_penalty = (preds.std() - targets.std()) ** 2
    full_loss = avg_val_loss + 1.0 * std_penalty.item()

    return full_loss

def main():
    print("\n--- Tuning KANGRU for exchange rate volatility forecasting ---")

    def objective(trial):
        try:
            return train_and_evaluate(trial)
        except Exception as e:
            print(f"Trial failed: {e}")
            return float("inf")

    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=SEED)
    )
    study.optimize(objective, n_trials=200)

    print("\n--- Best Trial ---")
    print(f"Trial #{study.best_trial.number}")
    print(f"Validation Loss: {study.best_value:.6f}")
    for key, value in study.best_trial.params.items():
        print(f"{key}: {value}")

    with open("best_kangru_exchange.txt", "a") as f:
        f.write(f"Trial #{study.best_trial.number}\n")
        f.write(f"Validation Loss: {study.best_value:.6f}\n")
        for key, value in study.best_trial.params.items():
            f.write(f"{key}: {value}\n")
        f.write("\n")

if __name__ == "__main__":
    main()
