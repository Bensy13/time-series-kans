import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import torch
from utils.stock_data import get_data_loaders, StockIndex
from models.kangru import KANGRU  # Adjust as needed
import optuna
from optuna.trial import Trial

SEED=42

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def train_and_evaluate(trial: Trial, stock: StockIndex):
    window_size = 7
    hidden_dim = trial.suggest_int("hidden_dim", 16, 128, step=16)
    batch_size = trial.suggest_int("batch_size", 16, 64, step=16)
    num_layers = trial.suggest_int("num_layers", 1, 3)
    grid = trial.suggest_int("grid", 5, 17)
    k = trial.suggest_int("k", 2, 5)
    lr = trial.suggest_float("lr", 1e-4, 1e-1, log=True)
    lamb = trial.suggest_float("lambda_reg", 1e-5, 5e-3, log=True)
    lamb_entropy = trial.suggest_float("lamb_entropy", 0.1, 2.0)
    lamb_l1 = trial.suggest_float("lamb_l1", 0.1, 2.0)
    lamb_coef = trial.suggest_float("lamb_coef", 1e-5, 1e-2, log=True)
    lamb_coefdiff = trial.suggest_float("lamb_coefdiff", 1e-5, 1e-2, log=True)

    train_dl, val_dl, _, _ = get_data_loaders(stock, window_size, batch_size)
    input_dim = next(iter(train_dl))[0].shape[-1]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = KANGRU(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        output_dim=1,
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
        patience=10,
        min_delta=1e-5,
        log=10
    )

    # Post-training check: penalize if predicted variance is too low
    model.eval()
    preds = []
    targets = []
    with torch.no_grad():
        for xb, yb, *_ in val_dl:
            xb = xb.to(device)
            y_pred = model(xb).squeeze()
            preds.append(y_pred.cpu())
            targets.append(yb)
    
    preds = torch.cat(preds, dim=0)
    targets = torch.cat(targets, dim=0)
    
    std_pred = torch.std(preds).item()
    std_true = torch.std(targets).item()
    
    # Penalize deviation from actual std
    std_penalty = (std_pred - std_true) ** 2
    return model.best_val_loss + 10. * std_penalty  # scale if needed


def main():
    stock = StockIndex.NVDA
    print(f"\n--- Tuning KANGRU for {stock.name} ---")

    def objective(trial):
        try:
            return train_and_evaluate(trial, stock)
        except Exception as e:
            print(f"Trial failed: {e}")
            return float("inf")

    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=42)
    )
    study.optimize(objective, n_trials=150)

    print(f"Best result for {stock.name}:")
    print(f"Trial #{study.best_trial.number}")
    print(f"Validation Loss: {study.best_value:.6f}")
    for key, value in study.best_trial.params.items():
        print(f"{key}: {value}")

    # Save best params to file
    with open(f"best_kangru_params.txt {stock.value}", "a") as f:
        f.write(f"=== {stock.name} ===\n")
        f.write(f"Trial #{study.best_trial.number}\n")
        f.write(f"Validation Loss: {study.best_value:.6f}\n")
        for key, value in study.best_trial.params.items():
            f.write(f"{key}: {value}\n")
        f.write("\n")


if __name__ == "__main__":
    main()
