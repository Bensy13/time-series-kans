import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import os
# Hardcoded parquet path
_PARQUET_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "exchange_rate.parquet"))

class VolatilityForecastDataset(Dataset):
    def __init__(self, df, window_size=10, horizon=3, standardize=True):
        self.window_size = window_size
        self.horizon = horizon
        self.standardize = standardize

        arr = df.values
        X, y = [], []

        for i in range(len(arr) - window_size - horizon):
            window = arr[i : i + window_size]
            future_vol = np.std(arr[i + window_size : i + window_size + horizon], axis=0)
            X.append(window)
            y.append(future_vol)

        X = np.array(X)
        y = np.array(y)

        if standardize:
            self.y_mean = y.mean(axis=0)
            self.y_std = y.std(axis=0) + 1e-6
            y = (y - self.y_mean) / self.y_std
        else:
            self.y_mean = None
            self.y_std = None

        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

    def inverse_transform(self, y_tensor):
        if self.standardize:
            return y_tensor * torch.tensor(self.y_std, device=y_tensor.device) + torch.tensor(self.y_mean, device=y_tensor.device)
        else:
            return y_tensor

def get_loaders(window_size=10, horizon=3, batch_size=64, standardize=True):
    df = pd.read_parquet(_PARQUET_PATH)
    log_returns = np.log(df / df.shift(1)).dropna()

    n = len(log_returns)
    train_df = log_returns.iloc[:int(0.8 * n)]
    val_df   = log_returns.iloc[int(0.8 * n):int(0.9 * n)]
    test_df  = log_returns.iloc[int(0.9 * n):]

    train_ds = VolatilityForecastDataset(train_df, window_size, horizon, standardize)
    val_ds   = VolatilityForecastDataset(val_df, window_size, horizon, standardize)
    test_ds  = VolatilityForecastDataset(test_df, window_size, horizon, standardize)

    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_dl   = DataLoader(val_ds, batch_size=batch_size)
    test_dl  = DataLoader(test_ds, batch_size=batch_size)

    return train_dl, val_dl, test_dl, train_ds, val_ds, test_ds
