# stock_picker

This is a side project in its early stages. Right now most of the code lives in
the branch `initial_modules` and is just an outline of an initial model (XGBoost).

Other models will be built to compare the output. The original idea was to test
the viability of an XGBoost regressor for time series forecasting when creating
features that make use of lags to introduce the model to past performance. Stocks
were chosen because they're a readily-available source of historical data.

Other models to be tried include:
- prophet
- ARIMA
- LSTM

Those will be added in subsequent updates..