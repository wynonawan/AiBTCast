import re
import numpy as np
import lightgbm as lgb
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import matplotlib.dates as mdates

from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.linear_model import LinearRegression
from linearmodels.panel import FamaMacBeth
from math import comb

btc_1yr = yf.download("BTC-USD", period="1y", interval="1h")
btc_data = btc_1yr[['Close', 'Volume']].dropna()

print(btc_1yr['Close'].tail(10))
print(btc_1yr['Volume'].tail(10))


# get CFO factor
def get_CFO(d, period=10):
    prices = d['Close'].values
    cfo = []
    for i in range(len(prices)):
        if i >= period - 1:
            Y = prices[i-period+1:i+1]
            X = np.arange(period).reshape(-1,1)
            model = LinearRegression().fit(X,Y)
            prediction = model.predict(np.array([[period-1]]))[0]
            cfo.append(100 * (prices[i] - prediction) / prices[i])
        else:
            cfo.append(np.nan)
    d['CFO'] = cfo
    return d

# get PWMA factor
def get_PWMA(d, n=10):
    prices  = d['Close']
    weights = np.array([comb(n-1,k) for k in range(n)])
    weights = weights / weights.sum()

    def apply_PWMA(window):
        return np.sum(window * weights)

    d['PWMA'] = prices.rolling(window=n).apply(apply_PWMA, raw=True)
    return d


# set up engineering data for training 

d = btc_data.copy()

d['P_1h'] = d['Close'].pct_change(1)
d['P_7h'] = d['Close'].pct_change(7)

d['V_1h'] = d['Volume'].pct_change(1)
d['V_7h'] = d['Volume'].pct_change(7)

d['PV_1h'] = d['P_1h'] * d['V_1h']
d['PV_7h'] = d['P_7h'] * d['V_7h']

d['vol']   = d['P_1h'].rolling(7).std()
d['ratio'] = d['Volume'] / d['Close']

d['Close_lag1'] = d['Close'].shift(1)

d = get_CFO(d, period=10)
d['CFO'] = d['CFO'].apply(lambda x: float(x) if x is not None else np.nan)

d = d.dropna()

X = d.drop(columns=['Close'])
Y = d['Close']

rows = int(0.9 * len(X))
X_train, X_val, Y_train, Y_val = X.iloc[:rows], X.iloc[rows:], Y.iloc[:rows], Y.iloc[rows:]

def clean_naming(data):
    data = data.copy()
    data.columns = [re.sub(r'[^0-9a-zA-Z_]', '_', str(c)) for c in data.columns]
    return data

lgb_params = {
    'objective': 'regression',
    'metric': 'mae',
    'boosting_type': 'gbdt', 
    'learning_rate': 0.11,
    'num_leaves': 100,
    'max_depth': 10,
    'feature_fraction': 0.8, 
    'bagging_fraction': 0.7, 
    'bagging_freq': 3,
    'lambda_l1': 1.0,
    'lambda_l2': 1.0,
    'verbose': -1
}

# training process
X_train = clean_naming(X_train)
X_val   = clean_naming(X_val)

d_train = lgb.Dataset(X_train, label=Y_train)
d_val   = lgb.Dataset(X_val, label=Y_val)

model = lgb.train(
    params=lgb_params,
    train_set=d_train,
    valid_sets=[d_train, d_val],
    num_boost_round=1500,
    callbacks=[lgb.early_stopping(stopping_rounds=100), lgb.log_evaluation(period=100)]
)

prediction_train = model.predict(X_train)
prediction_val = model.predict(X_val)

rmse_train = np.sqrt(mean_squared_error(Y_train, prediction_train))
rmse_val   = np.sqrt(mean_squared_error(Y_val,   prediction_val))

mae_train = mean_absolute_error(Y_train, prediction_train)
mae_val   = mean_absolute_error(Y_val, prediction_val)

tolerance = 0.01
accuracy = np.mean(np.abs((prediction_val - Y_val.values)/Y_val.values) < tolerance) * 100
#relative_error = np.mean(np.abs(prediction_val - Y_val.values) / Y_val.values) * 100
relative_error = mae_val / np.mean(Y_val.values) * 100

print(f"Training MAE: {mae_train:.2f}, RMSE: {rmse_train:.2f}")
print(f"Validation MAE: {mae_val:.2f}, RMSE: {rmse_val:.2f}")

print("actual price", Y_val.values[0:10])
print("predicted price", prediction_val[0:10])

pred_full = model.predict(X_val, num_iteration=model.best_iteration)

# select last month for plotting
start_date = Y_val.index[-1] - pd.Timedelta(days=30)
mask = Y_val.index >= start_date
Y_last  = Y_val.loc[mask]
pred_last = pred_full[mask]

next_pred = model.predict(X.iloc[-1:], num_iteration=model.best_iteration)
print("Predicted next hour BTC price:", next_pred[0])

next_time = Y_val.index[-1] + (Y_val.index[-1] - Y_val.index[-2]) 

plt.figure(figsize=(12,6))
plt.plot(Y_last.index, Y_last.values, label="Actual Price")
plt.plot(Y_last.index, pred_last, label="Predicted Price")
plt.scatter(next_time, next_pred[0], color='red', label=f"Next Hour Prediction: {next_pred[0]:.2f}", zorder=5)
plt.xlabel("Datetime")
plt.ylabel("BTC Price (USD)")
plt.title("Bitcoin Price Prediction: Last Month")
plt.legend()
plt.xticks(rotation=45)

#metrics_text = f"RMSE: {rmse_val:.2f}\nMAE: {mae_val:.2f}\nAccuracy (±1%): {accuracy:.2f}%"
metrics_text = f"RMSE: {rmse_val:.2f}\nMAE: {mae_val:.2f}\nMean Rel Error: {relative_error:.2f}%"

#plt.text(Y_val.index[0], max(Y_val.values)*1.01, metrics_text, fontsize=10, verticalalignment='top')
plt.text(Y_val.index[-1], max(Y_val.values)*1.01, metrics_text, fontsize=10, verticalalignment='top')
plt.tight_layout()


plt.savefig("bitcoin_prediction.png", dpi=300, bbox_inches='tight')

# get TC value
def TC_Rsquared(d, n):
    btc_values = d['Close'].values
    N = len(btc_values)
    tc = np.full(N, np.nan)

    for t in range(12*n, N-n):
        Y = btc_values[t-12*n:t-n]
        if len(tc) >= 4*n:
            X = np.arange(len(Y)).reshape(-1, 1)
            model = LinearRegression().fit(X, Y)
            R_2 = model.score(X,Y)
            tc[t] = R_2
    d['TC'] = tc
    return d


def add_factors(d, n):
    d_asset = d.copy()
    d_aseet = TC_Rsquared(d_asset, n)
    d_asset = get_PWMA(d_asset, n=10)
    d_asset = get_CFO(d_asset, period=10)
    d_asset['Return'] = d_asset['Close'].pct_change()
    return d_asset

## add factors to dataset
#n = 3
#d = TC_Rsquared(d, n)
#d = get_PWMA(d)
#d = get_CFO(d)

#assets_names = ["BTC-USD","ETH-USD","SOL-USD","BNB-USD"]
assets_names = [
    "BTC-USD","ETH-USD","SOL-USD","BNB-USD","ADA-USD","XRP-USD","DOGE-USD","DOT-USD",
    "AVAX-USD","MATIC-USD","LTC-USD","LINK-USD","ATOM-USD","NEAR-USD","FTM-USD",
    "ALGO-USD","TRX-USD","XLM-USD","SHIB-USD","ICP-USD"
]
assets_data = yf.download(assets_names, period="1y", interval="1d")

prices = assets_data['Close']
volume = assets_data['Volume']

assets = prices.stack().reset_index()
assets.columns = ["Datetime","Ticker","Close"]

assets_vol = volume.stack().reset_index()
assets_vol.columns = ["Datetime","Ticker","Volume"]
assets = pd.merge(assets, assets_vol, on=["Datetime","Ticker"])
assets = assets.set_index(["Ticker","Datetime"]).sort_index()

assets_factors = assets.groupby(level="Ticker", group_keys=False).apply(add_factors,n=3)


valid_assets = assets_factors[['Return','TC','PWMA','CFO']].dropna()
Y = valid_assets['Return']
X = sm.add_constant(valid_assets[['TC','PWMA','CFO']])

FamaM = FamaMacBeth(Y, X)
fm_result = FamaM.fit(cov_type="kernel")

print("Fama–MacBeth regression result:")
print(fm_result.summary)


