# Reliance Industries Stock Price Prediction

Predict whether Reliance Industries' (RELIANCE.NS) closing price **5 trading days from now** will be **strictly higher** than today's closing price, using a progression of machine learning models.

## Dataset

- **Source:** Yahoo Finance (via `yfinance`, `auto_adjust=True` for split/dividend-adjusted prices)
- **Ticker:** `RELIANCE.NS`
- **Period:** January 1, 2011 – January 1, 2024 (13 years of daily OHLCV data)
- **Total Trading Days:** 3,204

| Split | Date Range | Duration | Trading Days |
|-------|-----------|----------|-------------|
| Train | Jan 2011 – Dec 2020 | 10 years | 2,463 |
| Test  | Jan 2021 – Jan 2024 | 3 years | 741 |

## Features (22 total)

| Category | Features | Rationale |
|----------|----------|-----------|
| **Price Returns** | `ret_1`, `ret_5`, `ret_10`, `ret_20` | Capture momentum at multiple horizons |
| **Moving Average Ratios** | `ma5_ratio`, `ma10_ratio`, `ma20_ratio`, `ma50_ratio` | Scale-independent trend position (ratios instead of raw MAs) |
| **Volatility** | `vol_5`, `vol_20` | Indicate calm vs chaotic market regimes |
| **RSI** | `rsi_14` | Distinguishes sustainable rallies from overextended moves (>70 overbought, <30 oversold) |
| **Volume Changes** | `vol_change_1`, `vol_change_5` | Price rises on increasing volume are more sustainable |
| **Rolling Z-Scores** | `zscore_20`, `zscore_50` | Combines price deviation and volatility normalization into a single feature |
| **MACD** | `macd`, `macd_signal`, `macd_hist` | Detects trend momentum changes via EMA(12) vs EMA(26) crossovers |
| **Bollinger Bands** | `bb_upper_ratio`, `bb_lower_ratio`, `bb_bandwidth` | Price position relative to a volatility envelope around the 20-day MA |
| **ATR** | `atr_14` | Captures intraday volatility that rolling std of returns misses, normalized by price |

### Features Tested but Excluded
- **Volume Ratio** (daily volume / 20-day rolling mean volume): degraded model performance on the validation set.
- **RSI-7** (shorter-window RSI): introduced noise rather than signal for the 5-day prediction horizon.

## Target Variable

- **1** (Up) — Close(T+5) > Close(T)
- **0** (Down/Flat) — Close(T+5) ≤ Close(T)

## Models Tried

### 1. Logistic Regression (Baseline)
- **Test Accuracy:** 52.02%
- Linear model that could not capture the non-linear complexities of technical indicators.
- Served as a baseline to benchmark more sophisticated approaches.

### 2. Random Forest
- **Baseline (hand-tuned):** 55.5% test accuracy, severe overfitting (87.6% train vs 55.5% test).
- **Tuned with RandomizedSearchCV:** 55.78% test accuracy, 50.4% win rate, +46.66% total return, Sharpe 0.6162.
- **Tuned with Optuna (Final):** 56.07% test accuracy, 55.7% win rate, **+88.38% total return**, **Sharpe 1.0949**.

### 3. XGBoost (Optuna-tuned)
- Gradient boosting with L1/L2 regularization, learning rate control, and Optuna Bayesian optimization.
- Addressed Random Forest's overfitting problem but did not outperform the Optuna-tuned Random Forest in financial metrics.

### 4. Scipy-Optimized Stacking Ensemble (XGBoost + Random Forest + Logistic Regression)
- Custom `ScipyOptimizedEnsemble` class that strictly prevents look-ahead bias using chronological `TimeSeriesSplit`.
- `scipy.optimize.minimize` (SLSQP) discovers optimal voting weights that minimize Out-Of-Fold Log-Loss.
- Hyperparameters of all base estimators tuned simultaneously via Optuna.
- Despite achieving comparable classification accuracy, the ensemble exhibited a **Long/Short imbalance (105/10)**, revealing a directional bias that hurt risk-adjusted performance.

## Final Model Selection

**Random Forest tuned with Optuna** was selected as the final model because:
- It outperformed all other models in both financial metrics and signal quality.
- It produced a **balanced Long/Short ratio (72/43)**, avoiding the directional bias seen in the ensemble (105/10).
- Despite the ensemble achieving comparable classification accuracy, Random Forest's balanced signal generation produced **superior risk-adjusted returns**.

### Optuna vs RandomizedSearchCV
Both tuning methods were evaluated for Random Forest. While RandomizedSearchCV produced marginally higher classification accuracy, **Optuna's Bayesian optimization yielded significantly better financial performance**:

| Metric | RandomizedSearchCV | Optuna |
|--------|--------------------|--------|
| Total Return | +46.66% | **+88.38%** |
| Sharpe Ratio | 0.6162 | **1.0949** |
| Win Rate | 50.4% | **55.7%** |

Since the ultimate objective is a profitable trading strategy rather than classification accuracy alone, Optuna was adopted as the final tuning methodology.

## Trading Simulation (Test Set: 2021–2024)

| Metric | RF (RandomizedSearchCV) | RF (Optuna) ★ |
|--------|------------------------|---------------|
| Initial Capital | ₹10,00,000 | ₹10,00,000 |
| Final Capital | ₹14,66,570 | **₹18,83,752** |
| Total Return | +46.66% | **+88.38%** |
| Sharpe Ratio | 0.6162 | **1.0949** | TBD |
| Total Trades | 115 | 115 |
| Long / Short | 70 / 45 | 72 / 43 |
| Win Rate | 50.4% | **55.7%** |

★ = Final selected model

**Trading Rules:** Signal 1 → Long at Open(T+1). Signal 0 → Short at Open(T+1). Hold 5 days, exit at Close(T+5). 100% capital allocation, zero transaction costs.

## Feature Analysis

The feature importance analysis reveals clear hierarchical patterns:

1. **Volatility features dominate:** `vol_20` (20-day realized volatility) and `atr_14` (Average True Range) ranked highest, suggesting that the prevailing volatility regime is the strongest predictor of 5-day return direction for Reliance. `vol_5` also ranked in the top five.

2. **Trend and momentum features contribute significantly in the mid-tier:** All three MACD components appear in the top half, confirming that momentum regime changes carry meaningful predictive information. Medium-term returns (`ret_20`, `ret_10`) outperformed short-term returns (`ret_1`, `ret_5`), suggesting the model captures mean-reversion over longer lookback windows more effectively than immediate price action.

3. **Volume features ranked lowest:** `vol_change_1` and `vol_change_5` carry limited incremental signal beyond what price-based features already capture for this stock.

## Approaches Considered & Rejected

During the modeling and ensemble phases, several standard approaches were implemented but ultimately rejected:

- **`RandomizedSearchCV` / `GridSearchCV`:** Standard hyperparameter tuning methods were initially considered but discarded. **Optuna** uses Bayesian optimization (TPE) which learns from past trials to focus on high-yield parameter spaces rather than blindly searching. Optuna found hyperparameters that produced nearly **double the total return**.

- **Single Model Deployments:**
  - **Logistic Regression:** Rejected as a standalone because it could not capture non-linear patterns in technical indicators.
  - **Random Forest (hand-tuned):** Severe overfitting (87.6% train vs 55.5% test).
  - **Standalone XGBoost:** Good regularization properties but did not match Optuna-tuned RF in financial metrics.

- **Scikit-learn's `StackingClassifier`:** Relies on `cross_val_predict`, which strictly requires a data partition. It errors when given `TimeSeriesSplit` (the initial chronological fold is never a test set). Using `KFold`/`StratifiedKFold` instead would introduce **look-ahead bias**. Rejected in favor of the custom `ScipyOptimizedEnsemble` class.

- **Meta-Learners (`RidgeClassifier` / `LogisticRegression`):** Standard stacking trains a secondary model on base predictions. Financial data is extremely noisy, making a secondary model prone to overfitting OOF predictions. Rejected in favor of **`scipy.optimize.minimize` (SLSQP)** for explicit, constrained weight optimization.

- **Scipy Ensemble (final ensemble):** Despite comparable classification accuracy, its Long/Short imbalance (105/10) revealed directional bias. Random Forest's balanced signals (72/43) produced superior risk-adjusted returns.

## Requirements

```
numpy
pandas
yfinance
scikit-learn
xgboost
optuna
scipy
matplotlib
```

Install with:
```bash
pip install numpy pandas yfinance scikit-learn xgboost optuna scipy matplotlib
```

## Project Structure

```
Reliance-Stock-Forecast/
├── RandomF.ipynb                      # Random Forest baseline notebook
├── RandomF_Optuna.ipynb               # Random Forest + Optuna tuning (★ final model)
├── XGBoost.ipynb                      # XGBoost + Optuna tuning notebook
├── Ensemble_Scipy.ipynb               # Stacking Ensemble with Scipy Optimization
└── README.md
```
