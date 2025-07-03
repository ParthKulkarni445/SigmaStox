# 📘 SigmaStox: Portfolio Volatility Prediction

## ✨ Overview
SigmaStox is a two-stage machine-learning pipeline that first forecasts individual stock implied volatility (IV) and then aggregates these forecasts to predict portfolio-level IV. Stage I employs LSTM networks on historical OHLC and IV data to capture temporal patterns in each asset’s volatility. Stage II uses an MLP to combine individual IV predictions, portfolio weights, and correlation structures to estimate overall portfolio volatility. This modular approach provides granular insights at the stock level and holistic risk estimates at the portfolio level.

## 📚 Prerequisites (Theory)
- **Time Series Analysis:** Concepts of stationarity, autocorrelation, and basic ARIMA models; understanding how past observations inform future predictions.
- **Neural Network Fundamentals:** Structure and training of RNNs and LSTMs, including gate mechanisms and backpropagation through time; feedforward MLPs.
- **Optimization & Regularization:** Gradient descent and variants (e.g., Adam), learning rate schedules, dropout, and strategies to mitigate overfitting.
- **Financial Volatility Concepts:**  
  - **Realized Volatility (RV):**  
    \[\sigma_{\mathrm{RV},t} = \sqrt{\frac{1}{n-1} \sum_{i=1}^n (r_{t-i} - \bar r)^2}\]  
  - **Implied Volatility (IV):** Market’s expectation embedded in option prices, derived via the Black–Scholes model.
- **Portfolio Theory & Correlation:**  
  - **Correlation Coefficient:** \(\rho_{ij} = \frac{\mathrm{Cov}(r_i, r_j)}{\sigma_i \sigma_j}\)  
  - **Portfolio Variance:**  
    \[\sigma_p^2 = \sum_i w_i^2 \sigma_i^2 + 2 \sum_{i<j} w_i w_j \sigma_i \sigma_j \rho_{ij}\]  
  - **Portfolio IV:** \(\sigma_p = \sqrt{\sigma_p^2}\)
- **Statistical Evaluation Metrics:** Definitions of MSE, MAE, RMSE, and Pearson correlation coefficient for model assessment.

---

## 🎯 Problem Statement
Forecasting implied volatility accurately is critical for options pricing, portfolio risk management, and strategic trading. Traditional approaches often rely on historical realized volatility or simplistic models that fail to capture dynamic market expectations. SigmaStox addresses this by:

- **Stage I:** Leveraging LSTM networks to model each stock’s IV time series, capturing temporal dependencies and market nuances.  
- **Stage II:** Integrating individual IV forecasts with portfolio weights and correlation structures via an MLP to predict holistic portfolio IV.  

The goal is a robust, end‑to‑end pipeline that offers both granular stock‑level insights and comprehensive portfolio‑level risk estimates.

---

## 💾 Input Data
SigmaStox ingests historical market data via `yfinance` for a predefined universe of equities and, optionally, market indices (e.g., SPY, ^VIX). For each asset, the following daily fields are collected over a specified date range:

- **Open, High, Low, Close prices**: raw price levels for volatility feature engineering  
- **Volume**: trading volume for liquidity insights (optional)  
- **Implied Volatility (IV)**: computed as the average of call & put IVs from the nearest‑expiry option chain  

Data is organized into a tidy DataFrame with columns:  
`[date, ticker, open, high, low, close, volume (opt), iv]`.  
Missing IVs (e.g., illiquid days) are dropped to ensure clean model inputs.

---

## 🛠️ Methodology

### Stage I: Individual Stock IV Forecasting
1. **Feature Construction**  
   - Normalize OHLC by rolling means or previous close to remove scale effects.  
   - Compute additional indicators (e.g., rolling realized volatility, ATR, momentum) as optional inputs.

2. **Model Architecture**  
   - **LSTM**: sequence length = `window` (e.g., 20 days), input size = number of features (e.g., 4 or 5), hidden layers & dropout tuned via Optuna.  
   - Output = next‑day IV forecast.

3. **Training & Validation**  
   - 70/30 date‑based split for hyperparameter tuning.  
   - **Rolling (stepwise) validation**: retrain up to each validation date and record MSE to assess stability over time.  
   - Metrics: MSE, MAE, RMSE, and Pearson correlation between predicted vs actual IV.

### Stage II: Portfolio IV Prediction
1. **Feature Aggregation**  
   - **Predicted IVs**: daily forecasts from Stage I for each asset in the portfolio.  
   - **Weights**: portfolio allocation vector (equal or custom).  
   - **Rolling Correlations**: pairwise correlations of recent returns to capture diversification effects.  
   - **Market Regime**: optional VIX or index vol as a regime indicator.

2. **Target Computation**  
   - **Portfolio variance**: \(\sigma_p^2 = \sum_i w_i^2 \sigma_i^2 + 2 \sum_{i<j} w_i w_j \sigma_i \sigma_j \rho_{ij}\)  
   - **Portfolio IV**: \(\sigma_p = \sqrt{\sigma_p^2}\)

3. **Model Architecture**  
   - **MLP**: input dimension = (#assets + #correlations + optional regimes), two hidden layers with dropout.  
   - Output = portfolio IV forecast.

4. **Training & Validation**  
   - Train/val split on dates or expanding window.  
   - Metrics: RMSE, MAE, and correlation to measure forecasting accuracy.

---

## 📈 Output Interpretation
- **Stage I Outputs:**  
  - Per‑stock forecast time series of predicted IV vs. actual IV.  
  - Loss curves over epochs and rolling‑validation MSE plot to visualize model stability.  
  - Summary metrics (MSE, MAE, RMSE, correlation) per stock.

- **Stage II Outputs:**  
  - Portfolio‑level IV forecast compared against realized portfolio IV (from actual returns).  
  - Training/validation loss curves, scatter plot of predicted vs actual portfolio IV.  
  - Key metrics (RMSE, MAE, Pearson r) to quantify model performance.

These outputs help you gauge both micro (stock‑level) and macro (portfolio‑level) forecasting accuracy, guiding model refinement and risk management decisions.
