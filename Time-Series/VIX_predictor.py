# import libraries
try:
  import arch
  from arch import arch_model
  from arch.__future__ import reindexing
except ModuleNotFoundError:
  !pip install arch
  import arch
  from arch import arch_model
  from arch.__future__ import reindexing

try:
  import yfinance as yf
except ModuleNotFoundError:
  !pip install yfinance
  import yfinance as yf

import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ticker = "^GSPC"
new_data = yf.download(ticker, start="2010-11-01", end="2022-06-01")
# use pct_change() to calculate percentage change between close positions
new_data["Return"] = (new_data["Close"].pct_change()) * 100
print(new_data)
# clean data by dropping empty data values
new_data.dropna(inplace=True)
# plot returns over time
fig = plt.figure()
fig.set_figwidth(20)
plt.plot(new_data["Return"], label="Daily Returns")
plt.legend(loc="upper right")
plt.title("Daily Returns Over Time")
plt.show()

daily_volatility = new_data["Return"].std()
print(f"Daily volatility: {daily_volatility}")

"""

After some quick googling (due to my lack of finance knowledge), a GARCH model seems most appropriate.
https://www.investopedia.com/terms/g/garch.asp

The parameters p and q represent the number of lag variances (aka. distributed lag) for the model and the lag residual errors respectively.
"""
garch_model = arch.arch_model(new_data["Return"], p=1, q=1, mean="constant", vol="GARCH", dist="normal")
apply_garch = garch_model.fit(disp="off")
# Summary of GARCH application
print(apply_garch.summary())
# Printing the parameters outputs a number of variables, we're most interested in the alpha. The bigger the alpha, the more volatile!
# On the other hand, the larger the beta, the longer the impact the volatility has
print(apply_garch.params)
# The horizon parameter dictates how many periods ahead we'd like to forecast
forecast = apply_garch.forecast(horizon = 5)
# Print forecast results inc. 1-step ahead forecast up to and including June 1st 2022
print(forecast.variance[-1:])

rolling_predictions = []
test_size = 365 # test on one year

for i in range(test_size):
    training_data = new_data["Return"][:-(test_size-i)]
    model = arch_model(training_data, p=1, q=1)
    model_fit = model.fit(disp='off')
    predict = model_fit.forecast(horizon=1)
    rolling_predictions.append(np.sqrt(predict.variance.values[-1,:][0]))
    
print("Training Data")
print(training_data)
rolling_predictions = pd.Series(rolling_predictions, index=new_data["Return"].index[-365:])

plt.figure(figsize=(10,4))
plt.plot(rolling_predictions)
plt.title('Rolling Prediction')
plt.show()
