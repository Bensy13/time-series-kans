# Time Series Forecasting with Kolmogorov–Arnold Networks

This repository contains code and experiments for a final-year project investigating Kolmogorov–Arnold Network (KAN) architectures for multivariate time series forecasting on financial data. The focus is on evaluating KANs in combination with recurrent models such as LSTMs and GRUs, comparing performance, interpretability, and computational cost.

---

## 📁 Repository Structure

### `data/`
Preprocessed datasets used for training and evaluation.
- `stock_data.parquet`: OHLCV data for NVDA, AAPL, KO from Yahoo Finance.
- `exchange_rate.parquet`: Supplementary exchange rate data.

### `models/`
Core model implementations.
- `kan_encoder.py`: KAN encoder for static feature transformation.
- `kangru.py`: Hybrid GRU with embedded spline-based transformations.
- `lstm.py`, `mlp_encoder.py`: Standard LSTM and MLP encoder baselines.

### `notebooks/`
Jupyter notebooks for training, analysis, and visualisation.

#### Log-Return Experiments (`notebooks/log_returns/`)
- `01-train-lstm-baseline.ipynb`: Baseline LSTM training.
- `02–03.5`: Variants with fixed/trainable MLP or KAN encoders to LSTM.
- `kan_gru.ipynb`: KANGRU evaluation.
- `timing.ipynb`: Runtime benchmarking.

#### Min-Max Experiments (`notebooks/minmax/`)
- Same structure as above but with min-max input scaling instead of log returns.
- Includes affine transform analysis (`affine-true/false` in filenames).

### `tuning/`
Standalone training scripts (non-notebook) for automated tuning and evaluation.
Organised by input preprocessing (`log_returns`, `minmax`, `exchange_test`).
- `kan_lstm.py`, `mlp_lstm.py`, `kan_gru.py`, etc.

### `best_params/`
Saved best results from tuning.
- `log_returns/`: Best results (e.g., `best_kangru_params.txt`, etc.)
- `minmax/`: Affine vs. non-affine comparisons across models.

### `utils/`
Utility scripts and data loaders.
- `fetch_stock_data.py`: Yahoo Finance scraper.
- `stock_data.py`: Preprocessing and data splitting.
- `exchange_data.py`: Exchange rate processing.
- `early_stopping.py`: Callback for training.

---

## 🔧 Requirements

Install dependencies with:
```bash
pip install -r requirements.txt
