# QC Task 1 - Reliance Industries Stock Price Prediction

Predict whether Reliance Industries' (RELIANCE.NS) closing price **5 trading days from now** will be **strictly higher** than today's closing price. The progression of models includes **Logistic Regression**, **Random Forest**, **XGBoost**, and finally a fully-tuned **Stacking Ensemble (XGBoost + Random Forest + Logistic Regression)** weighted optimally via **scipy.optimize**.

## Dataset

- **Source:** Yahoo Finance (via `yfinance`)
- **Ticker:** `RELIANCE.NS`
- **Period:** January 1, 2011 - January 1, 2024 (13 years of daily data)
- **Total Trading Days:** 3,204

| Split | Date Range | Duration | Trading Days |
|-------|-----------|----------|-------------|
| Train | Jan 2011 - Dec 2020 | 10 years | 2,463 |
| Test  | Jan 2021 - Jan 2024 | 3 years | 741 |

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

- **1** - Close(T+5) > Close(T) (price goes up)
- **0** - Close(T+5) <= Close(T) (price goes down or stays flat)

## Model Progression

1. **Logistic Regression** (baseline) - Linear model, 52% test accuracy. Could not capture non-linear patterns.
2. **Random Forest** - Non-linear ensemble, 55.5% test accuracy. Prone to overfitting (87.6% train vs 55.5% test).
3. **XGBoost (Optuna)** - Gradient boosting. Addressed RF's overfitting through L1/L2 regularization and learning rate control.
4. **Scipy-Optimized Stacking Ensemble** (Final) - A custom-built `TimeSeriesStacking` ensemble combining XGBoost, Random Forest, and Logistic Regression. 

### Custom Time-Series Stacking
To strictly prevent look-ahead bias, the ensemble drops scikit-learn's native `StackingClassifier` (which expects a randomized partition) for a custom implementation:
- **Phase 1**: Base learners generate Out-Of-Fold probabilities using a chronological `TimeSeriesSplit(n_splits=5)`.
- **Phase 2**: Instead of using a meta-learner (like Ridge/Logistic Regression), we use `scipy.optimize.minimize` (SLSQP method) to discover the literal voting weights that minimize Out-Of-Fold Log-Loss.
- **Phase 3**: Base learners re-train on the full dataset, and final inference is a weighted softmax vote.

Optuna wraps the entire workflow, tuning hyperparameters for all distinct base estimators simultaneously to find the perfectly synergized combination.

### Approaches Considered & Rejected
During the modeling and ensemble phases, several standard approaches were implemented but ultimately rejected in favor of the final solution:
- **`RandomizedSearchCV` / `GridSearchCV`**: Standard hyperparameter tuning methods were initially considered but discarded. **Optuna** uses Bayesian optimization (TPE) which learns from past trials to focus on high-yield parameter spaces rather than blindingly searching a grid or random sample. Optuna found better hyperparameters significantly faster.
- **Single Model Deployments**: 
  - **Logistic Regression**: Used as the initial baseline but rejected because it could not capture the non-linear complexities of the technical indicators.
  - **Random Forest**: Found severe overfitting (87.6% train vs 55.5% test accuracy) since individual trees tend to perfectly memorize noisy financial features.
  - **Standalone XGBoost**: Excellent standalone model with L1/L2 regularization to prevent the Random Forest overfitting, but rejected for production because ensembling it with other decorrelated models (RF, LR) offered a lower-variance, more robust overall strategy.
- **Scikit-learn's `StackingClassifier`**: Naive `StackingClassifier` relies on `cross_val_predict`, which strictly requires a partition of indices to generate out-of-fold features. It throws an error when given a `TimeSeriesSplit` (because the initial chronological fold is never used as a test set). Bypassing this by using a standard `KFold` or `StratifiedKFold` would introduce severe **look-ahead bias**, allowing base models to train on future data to predict the past. This was completely rejected in favor of our custom `TimeSeriesStacking` class.
- **Meta-Learners (`RidgeClassifier` / `LogisticRegression`)**: Standard stacking trains a secondary linear meta-learner on the base predictions. Financial data is extremely noisy, and fitting a secondary model can easily overfit the Out-Of-Fold predictions. We rejected fitting a meta-estimator in favor of using a **bounded mathematical optimization (`scipy.optimize.minimize` via SLSQP)** to discover explicit, non-negative soft-voting weights that strictly sum to 1. This proved much more robust and interpretable.

## Trading Simulation (Test Set: 2021-2024)

| Metric | Random Forest | XGBoost | Stacking Ensemble |
|--------|--------------|---------|-------------------|
| Initial Capital | Rs.10,00,000 | Rs.10,00,000 | Rs.10,00,000 |
| Final Capital | Rs.14,66,570 | **TBD** | **TBD** |
| Total Return | +46.66% | **TBD** | **TBD** |
| Sharpe Ratio | 0.6162 | **TBD** | **TBD** |
| Total Trades | 115 | **TBD** | **TBD** |
| Win Rate | 50.4% | **TBD** | **TBD** |

> **Note:** XGBoost and Stacking Ensemble metrics will be naturally populated down the notebook depending on the Optuna trial run.

**Trading Rules:** Signal 1 - Long at Open(T+1), Signal 0 - Short at Open(T+1). Hold 5 days, exit at Close(T+5). 100% capital allocation, zero transaction costs.

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
python-docx
```

Install with:
```bash
pip install numpy pandas yfinance scikit-learn xgboost optuna scipy matplotlib python-docx
```

## Usage

```bash
python load_reliance_data.py    # Logistic Regression
python random_forest.py         # Random Forest + Trading Simulation
```

For the Optuna-tuned Stacking Ensemble, run `Ensemble_main.ipynb` directly in Jupyter/VS Code.

## Project Structure

```
QC Task1/
├── load_reliance_data.py              # Logistic Regression pipeline
├── random_forest.py                   # Random Forest + trading simulation
├── Random_Forest.ipynb                # Random Forest notebook
├── XGBoost.ipynb                      # XGBoost + Optuna tuning notebook
├── Ensemble_main.ipynb                # Final Stacking Ensemble (XGB+RF+LR) with Scipy Optimization
├── QC_Task1_Methodology_Report.docx   # Research report
├── generate_report.py                 # Script to generate the .docx report
├── feature_importance.png             # RF feature importance chart
└── README.md
```
