# stock_picker

This is a side project in its early stages. Right now most of the code lives in
the branch `development` and is just an outline of an initial model (XGBoost) 
and a forecaster (SARIMAX).

Other models will be built to compare the output. The original idea was to test
the viability of an XGBoost regressor for time series forecasting when creating
features that make use of lags to introduce the model to past performance,
comparing the predictive accuracy of that with other more common forecasters.
Stocks were chosen because they're a readily-available source of historical 
data (`yfinance`).

Two models included thus far:
- XGBoost
- SARIMAX

Other models to be tried include:
- prophet
- LSTM

Those will be added in subsequent updates..