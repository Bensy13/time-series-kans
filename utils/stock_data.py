import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from enum import Enum

file_path = r"data/stock_data.parquet"


class StockIndex(Enum):
    NVDA = 'NVDA'
    AAPL = 'AAPL'
    KO = 'KO'

class TimeSeriesDataset(Dataset):
    def __init__(self, df: pd.DataFrame, stock: StockIndex, window_size: int):
        self.window_size = window_size
        self.stock = stock.value

        # Extract features
        self.stock_features = df[[ (self.stock, col) for col in ['Open', 'High', 'Low', 'Close', 'Volume'] ]]
        self.sp500_close = df[('^GSPC', 'Close')]
        self.dowjones_close = df[('^DJI', 'Close')]

        # Combine into one array
        self.data = pd.concat([self.stock_features, self.sp500_close, self.dowjones_close], axis=1).values
        self.length = len(self.data) - window_size

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        x = self.data[idx : idx + self.window_size]       # shape: (window, features)
        y = self.data[idx + self.window_size][3]          # target: stock Close price (position 3 in features)
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

def split_dataframe(df: pd.DataFrame, splits=(0.7, 0.1, 0.2)):
    assert sum(splits) == 1.0, "Splits must sum to 1.0"
    n = len(df)
    train_end = int(n * splits[0])
    val_end = train_end + int(n * splits[1])
    return df.iloc[:train_end], df.iloc[train_end:val_end], df.iloc[val_end:]

def get_data_loaders(stock: StockIndex, window_size: int, batch_size: int, splits=(0.7, 0.1, 0.2), shuffle_train=True):
    df = pd.read_parquet(file_path)

    # Time-based split
    df_train, df_val, df_test = split_dataframe(df, splits)

    # Create datasets
    train_ds = TimeSeriesDataset(df_train, stock, window_size)
    val_ds   = TimeSeriesDataset(df_val, stock, window_size)
    test_ds  = TimeSeriesDataset(df_test, stock, window_size)

    # DataLoaders
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=shuffle_train)
    val_dl   = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_dl  = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    return train_dl, val_dl, test_dl
