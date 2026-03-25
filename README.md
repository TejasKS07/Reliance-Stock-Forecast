# QC Task 1 — Reliance Industries Stock Price Prediction

Predict whether Reliance Industries' (RELIANCE.NS) closing price **5 trading days from now** will be **strictly higher** than today's closing price, using **Logistic Regression** and **Random Forest**.

## Dataset

- **Source:** Yahoo Finance (via `yfinance`)
- **Ticker:** `RELIANCE.NS`
- **Period:** January 1, 2011 → January 1, 2024 (13 years of daily data)
- **Total Trading Days:** 3,204

| Split | Date Range | Duration | Trading Days |
|-------|-----------|----------|-------------|
| Train | Jan 2011 – Dec 2020 | 10 years | 2,463 |
| Test  | Jan 2021 – Jan 2024 | 3 years | 741 |

## Features (22 total)

| Feature | Description |
|---------|------------|
| `ret_1`, `ret_5`, `ret_10`, `ret_20` | Price returns (momentum) |
| `ma5_ratio`, `ma10_ratio`, `ma20_ratio`, `ma50_ratio` | Price / Moving Average ratio (trend position) |
| `vol_5`, `vol_20` | Rolling std of returns (volatility) |
| `rsi_14` | 14-day Relative Strength Index |
| `vol_change_1`, `vol_change_5` | % change in trading volume |
| `zscore_20`, `zscore_50` | Rolling Z-scores (normalized deviation) |
| `macd`, `macd_signal`, `macd_hist` | MACD line, signal line, histogram |
| `bb_upper_ratio`, `bb_lower_ratio`, `bb_bandwidth` | Bollinger Bands (position & width) |
| `atr_14` | Average True Range (price volatility) |

## Target Variable

- **1** → Close(T+5) > Close(T) (price goes up)
- **0** → Close(T+5) ≤ Close(T) (price goes down or stays flat)

## Model Results (Test Set)

| Metric | Logistic Regression | Random Forest |
|--------|-------------------|--------------|
| Accuracy | 52.02% | **55.49%** |
| Up F1 | 0.61 | **0.61** |
| Down/Flat F1 | 0.37 | **0.47** |

## Trading Simulation (Test Set: 2021–2024)

| Metric | Value |
|--------|-------|
| Initial Capital | Rs.10,00,000 |
| **Final Capital** | **Rs.14,66,570** |
| **Total Return** | **+46.66%** |
| **Sharpe Ratio** | **0.6162** |
| Total Trades | 115 (70 Long / 45 Short) |
| Win Rate | 50.4% |

**Trading Rules:** Signal 1 → Long at Open(T+1), Signal 0 → Short at Open(T+1). Hold 5 days, exit at Close(T+5). 100% capital allocation, zero transaction costs.

## Requirements

```
numpy
pandas
yfinance
scikit-learn
python-docx
```

Install with:
```bash
pip install numpy pandas yfinance scikit-learn python-docx
```

## Usage

```bash
python load_reliance_data.py    # Logistic Regression
python random_forest.py         # Random Forest + Trading Simulation
```

## Project Structure

```
QC Task1/
├── load_reliance_data.py              # Logistic Regression (data + features + model)
├── random_forest.py                   # Random Forest + trading simulation
├── Random_Forest.ipynb                # Random Forest notebook
├── Log_Reg.ipynb                      # Logistic Regression notebook
├── QC_Task1.ipynb                     # Original notebook
├── QC_Task1_Methodology_Report.docx   # Research report
├── generate_report.py                 # Script to generate the .docx report
└── README.md
```
