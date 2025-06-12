import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from enum import Enum
from typing import Tuple, Dict
import os

# Path fix
file_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "stock_data.parquet"))

class StockIndex(Enum):
    NVDA = 'NVDA'
    AAPL = 'AAPL'
    KO = 'KO'

# Centralized feature config
STOCK_FEATURES = ['Open', 'High', 'Low', 'Close', 'Volume']
INDEX_FEATURES = ['^GSPC_Close', '^DJI_Close']
DATE_FEATURES = ['day_sin', 'day_cos']
ALL_FEATURES = STOCK_FEATURES + INDEX_FEATURES + DATE_FEATURES
INPUT_DIM = len(ALL_FEATURES)

class TimeSeriesDataset(Dataset):
    def __init__(self, df: pd.DataFrame, stock: StockIndex, window_size: int):
        self.window_size = window_size
        self.stock = stock.value

        self.data = df.copy()
        self.data.columns = ['_'.join(col) if isinstance(col, tuple) else col for col in self.data.columns]

        if 'day_sin' not in self.data.columns:
            self.data['day_of_year'] = self.data.index.dayofyear
            self.data['day_sin'] = np.sin(2 * np.pi * self.data['day_of_year'] / 365)
            self.data['day_cos'] = np.cos(2 * np.pi * self.data['day_of_year'] / 365)
        df.loc[:, 'day_sin'] = (df['day_sin'] + 1) / 2
        df.loc[:, 'day_cos'] = (df['day_cos'] + 1) / 2


        selected_columns = [f"{self.stock}_{f}" for f in STOCK_FEATURES] + INDEX_FEATURES + DATE_FEATURES
        self.data = self.data[selected_columns].copy()
        self.data = self.data.values
        self.length = len(self.data) - window_size

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        x = self.data[idx : idx + self.window_size]
        y = self.data[idx + self.window_size][3]  # 'Close' index in selected features
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

def split_dataframe(df: pd.DataFrame, splits=(0.7, 0.1, 0.2)):
    assert sum(splits) == 1.0, "Splits must sum to 1.0"
    n = len(df)
    train_end = int(n * splits[0])
    val_end = train_end + int(n * splits[1])
    return df.iloc[:train_end], df.iloc[train_end:val_end], df.iloc[val_end:]

def get_data_loaders(stock: StockIndex, window_size: int, batch_size: int, use_val_split=True, splits=(0.8, 0.1, 0.1), percentage_of_data=100):

    df = pd.read_parquet(file_path)

    # --- Data Cleaning and Feature Engineering ---
    df = df.astype({col: 'float64' for col in df.columns if isinstance(col, tuple) and col[1] == 'Volume'})
    df = df.ffill().dropna()
    df.columns = ['_'.join(col) if isinstance(col, tuple) else col for col in df.columns]
    df['day_of_year'] = df.index.dayofyear
    df['day_sin'] = np.sin(2 * np.pi * df['day_of_year'] / 365)
    df['day_cos'] = np.cos(2 * np.pi * df['day_of_year'] / 365)

    selected_features = [f"{stock.value}_{f}" for f in STOCK_FEATURES] + INDEX_FEATURES + DATE_FEATURES
    df = df[selected_features]

    if percentage_of_data < 100:
        n = int(len(df) * (percentage_of_data / 100))
        df = df.iloc[-n:]

    # --- Splitting ---
    df_train, df_val, df_test = split_dataframe(df, splits)

    # --- Scaling (Train + Val scaling logic split) ---
    scalers = {}

    if use_val_split:
        for col in df_train.columns:
            scaler = MinMaxScaler()
            scaler.fit(df_train[[col]])

            df_train.loc[:, col] = scaler.transform(df_train[[col]]).flatten()
            df_val.loc[:, col]   = scaler.transform(df_val[[col]]).flatten()
            df_test.loc[:, col]  = scaler.transform(df_test[[col]]).flatten()


            scalers[col] = scaler

        train_ds = TimeSeriesDataset(df_train, stock, window_size)
        val_ds   = TimeSeriesDataset(df_val, stock, window_size)
        test_ds  = TimeSeriesDataset(df_test, stock, window_size)

        train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        val_dl   = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
        test_dl  = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

        return train_dl, val_dl, test_dl, scalers

    else:
        # --- Refit scaler on full train+val if no validation split ---
        combined_df = pd.concat([df_train, df_val])
        for col in combined_df.columns:
            scaler = MinMaxScaler()
            scaler.fit(combined_df[[col]])

            combined_df.loc[:, col] = scaler.transform(combined_df[[col]]).flatten()
            df_test.loc[:, col]     = scaler.transform(df_test[[col]]).flatten()

            scalers[col] = scaler

        combined_ds = TimeSeriesDataset(combined_df, stock, window_size)
        test_ds     = TimeSeriesDataset(df_test, stock, window_size)

        combined_dl = DataLoader(combined_ds, batch_size=batch_size, shuffle=True)
        test_dl     = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

        return combined_dl, test_dl, scalers

class LogReturnDataset(Dataset):
    def __init__(self, df, stock: StockIndex, window_size: int):
        self.window_size = window_size
        self.stock = stock.value
        self.df = df.reset_index(drop=True)

        self.data = self.df.drop(columns=["Close_raw"]).values
        self.raw_close = self.df["Close_raw"].values
        self.length = len(self.data) - window_size

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        x = self.data[idx : idx + self.window_size]
        y = self.data[idx + self.window_size][STOCK_FEATURES.index("Close")]  # log return
        prev_close = self.raw_close[idx + self.window_size - 1]  # P_{t-1}
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32), torch.tensor(prev_close, dtype=torch.float32)

# class LogReturnDataset(Dataset):
#     def __init__(self, df, stock: StockIndex, window_size: int, horizon: int = 1):
#         self.window_size = window_size
#         self.horizon = horizon
#         self.stock = stock.value
#         self.df = df.reset_index(drop=True)

#         self.data = self.df.drop(columns=["Close_raw"]).values
#         self.raw_close = self.df["Close_raw"].values
#         self.length = len(self.data) - window_size - horizon + 1

#     def __len__(self):
#         return self.length

#     def __getitem__(self, idx):
#         x = self.data[idx : idx + self.window_size]
#         y = self.data[idx + self.window_size : idx + self.window_size + self.horizon, STOCK_FEATURES.index("Close")]
#         prev_close = self.raw_close[idx + self.window_size - 1]  # P_t

#         return (
#             torch.tensor(x, dtype=torch.float32),              # shape: [window_size, input_dim]
#             torch.tensor(y, dtype=torch.float32),              # shape: [horizon]
#             torch.tensor(prev_close, dtype=torch.float32),     # scalar
#         )



def get_log_return_loaders(stock: StockIndex, window_size: int, batch_size: int, splits=(0.8, 0.1, 0.1)):
    df = pd.read_parquet(file_path)

    # --- Clean and enrich ---
    df.columns = ['_'.join(col) if isinstance(col, tuple) else col for col in df.columns]
    df = df.ffill().dropna()
    df['day_of_year'] = df.index.dayofyear
    df['day_sin'] = np.sin(2 * np.pi * df['day_of_year'] / 365)
    df['day_cos'] = np.cos(2 * np.pi * df['day_of_year'] / 365)

    # --- Preserve raw Close for later reconstruction ---
    df['Close_raw'] = df[f"{stock.value}_Close"]

    # --- Compute log returns for stock features ---
    log_df = pd.DataFrame(index=df.index)
    for feat in STOCK_FEATURES:
        col = f"{stock.value}_{feat}"
        shifted = df[col].shift(1)
        valid = (shifted > 0) & (df[col] > 0)
        log_df[col] = np.log(df[col] / shifted).where(valid)

    # --- Compute log returns for index features ---
    for col in INDEX_FEATURES:
        shifted = df[col].shift(1)
        valid = (shifted > 0) & (df[col] > 0)
        log_df[col] = np.log(df[col] / shifted).where(valid)

    # --- Add day encodings + Close_raw target reference ---
    log_df['day_sin'] = df['day_sin']
    log_df['day_cos'] = df['day_cos']
    log_df['Close_raw'] = df['Close_raw']

    # --- Drop NaNs caused by shifting ---
    log_df.dropna(inplace=True)

    # --- Define column ordering for consistency ---
    input_cols = [f"{stock.value}_{f}" for f in STOCK_FEATURES] + INDEX_FEATURES + ['day_sin', 'day_cos']
    log_df = log_df[input_cols + ['Close_raw']]

    # --- Rescale sin/cos to match Close return range ---
    target_col = f"{stock.value}_Close"
    rmin, rmax = log_df[target_col].min(), log_df[target_col].max()
    for col in ['day_sin', 'day_cos']:
        log_df[col] = (log_df[col]) * (rmax - rmin) + rmin

    # --- Split ---
    train_df, val_df, test_df = split_dataframe(log_df, splits)

    # --- Create datasets and loaders ---
    train_ds = LogReturnDataset(train_df, stock, window_size)
    val_ds   = LogReturnDataset(val_df, stock, window_size)
    test_ds  = LogReturnDataset(test_df, stock, window_size)

    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_dl   = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_dl  = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    return train_dl, val_dl, test_dl


if __name__ == "__main__":

    stock = StockIndex.NVDA
    train_dl, val_dl, test_dl, scalers = get_data_loaders(stock, window_size=7, batch_size=32)
    print(f"Train batches: {len(train_dl)}, Val: {len(val_dl)}, Test: {len(test_dl)}")
    xb, yb = next(iter(train_dl))
    # View first sample in the first batch
    sample_x, sample_y = xb[0], yb[0]
    
    print("Sample X (10 timesteps × 9 features):")
    print(sample_x)
    
    print("Corresponding Y (target close price):")
    print(sample_y)

    print("Sample batch shape:", xb.shape, yb.shape)

    window_size = 10
    batch_size = 32

    train_dl, val_dl, test_dl = get_log_return_loaders(stock, window_size, batch_size)

    print(f"Train batches: {len(train_dl)}, Val: {len(val_dl)}, Test: {len(test_dl)}")

    xb, yb, p_tm1 = next(iter(test_dl))  # get first batch

    # View first sample in batch
    sample_x = xb[0]       # shape (window_size, num_features)
    sample_y = yb[0].item()       # scalar log return target
    sample_prev_price = p_tm1[0].item()  # scalar P_{t-1}

    print("Sample X (10 timesteps × N features):")
    print(sample_x)

    print("\nCorresponding Y (target log return):")
    print(sample_y)

    print("\nPrevious Close price (P_t-1):")
    print(sample_prev_price)

    print("\nBatch shapes:")
    print("X:", xb.shape, "| Y:", yb.shape, "| P_t-1:", p_tm1.shape)
