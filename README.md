# Reliance Industries Stock Price Prediction

Logistic Regression model to predict whether Reliance Industries' (RELIANCE.NS) closing price **5 trading days from now** will be **strictly higher** than today's closing price.

## Dataset

- **Source:** Yahoo Finance (via `yfinance`)
- **Ticker:** `RELIANCE.NS`
- **Period:** January 1, 2011 → January 1, 2024 (13 years of daily data)
- **Total Trading Days:** 3,204

| Split | Date Range | Trading Days |
|-------|-----------|-------------|
| Train | Jan 2011 – Dec 2020 | 2,463 |
| Test  | Jan 2021 – Jan 2024 | 741 |

## Features

| Feature | Description |
|---------|------------|
| `ret_1`, `ret_5`, `ret_10`, `ret_20` | Price returns over 1, 5, 10, 20 days (momentum) |
| `ma5_ratio`, `ma10_ratio`, `ma20_ratio`, `ma50_ratio` | Price / Moving Average ratio (trend position) |
| `vol_5`, `vol_20` | Rolling standard deviation of returns (volatility) |
| `rsi_14` | 14-day Relative Strength Index (overbought/oversold) |
| `vol_change_1`, `vol_change_5` | % change in trading volume (activity) |

## Target Variable

Binary classification:
- **1** → Close(T+5) > Close(T) (price goes up)
- **0** → Close(T+5) ≤ Close(T) (price goes down or stays flat)

## Results

| Metric | Train | Test |
|--------|-------|------|
| Accuracy | 54.23% | 52.89% |
| Up Precision | 0.55 | 0.54 |
| Up Recall | 0.77 | 0.73 |

## Requirements

```
numpy
pandas
yfinance
scikit-learn
```

Install with:
```bash
pip install numpy pandas yfinance scikit-learn
```

## Usage

```bash
python load_reliance_data.py
```

## Project Structure

```
QC Task1/
├── load_reliance_data.py   # Main script (data download, features, model)
├── QC_Task1.ipynb          # Jupyter notebook
└── README.md
```
