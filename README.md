# AiCryptoCast
AiCryptoCast is a Python project that predicts the real-time next-hour Bitcoin (BTC-USD) price using LightGBM and various engineered features: price, volume, and CFO (Cumulative Forecast Oscillator). Such engineering also performs well for next-4-hour prediction, which will be used in comparison in this repo.
In addition, this model also includes factor extraction for TC (Trend Confidence), PWMA (Predictive Weighted Moving Average), incorporating the FamaMacBeth estimation for all factors across multiple crypto assets. THe model reaches decent accuracy with ~0.25% overall error calculated from MAE, and ~0.34% from RMSE; it successfully captures the future movement trend.
This project is built for personal interest

## Contents
- [Install](https://github.com/wynonawan/AiCryptoCast/blob/main/README.md#installation-steps)
- Bitcoin Forcast:
  - [Next-hour bitcoin price](https://github.com/wynonawan/AiCryptoCast/blob/main/README.md#1-next-hour-bitcoin-price)
  - [Next-4-hour bitcoin price](https://github.com/wynonawan/AiCryptoCast/blob/main/README.md#2-next-4-hour-bitcoin-price)
  - [TC, PWMA, CFO factors](https://github.com/wynonawan/AiCryptoCast/blob/main/README.md#3-tc-pwma-and-cfo-factors)
  - [FamaMacBeth Estimation across cryptos]()

## Installation Steps
### 1. Get this repository by running:
```
git clone https://github.com/wynonawan/AiCryptoCast
```


### 2. Most libraries run in python3.9 or above. If don't have the upgraded version, you can upgrade your python by running:
```
brew install python@3.9
```
   Or, you can create a virtual python3.9 environment and run your code in it, without changing your local python package version, 
```
python3.9 -m venv venv
source venv/bin/activate
```
If you use conda for virtual environment already, then run
```
conda create -n quantenv python=3.9
conda activate quantenv
```


### 3. Made sure your `pip` is updated:

```
pip install --upgrade pip
```


### 4. Install all the python libraries necessary to run the codes (if you don't have so already)

```
pip install pandas numpy matplotlib yfinance scikit-learn lightgbm statsmodels linearmodels

```

## * BitCoin Forecast

### 1. Next-hour bitcoin price

This model implements LightGBM machine learning method effective in crypto data training in the quantitative field [see reference paper](https://www.sciencedirect.com/science/article/pii/S1057521923005719#:~:text=.%2C%202022). The initial training only included bitcoin features regarding price, volumn and volatility. The backtesting using Fama-MachBeth estimation has indicated the significance of CFO factor in describing the current bitcoin trend, hence an extra engineering feature -- CFO -- was as added to further improve the ML model. 

Run command:
```
python bitcoin_forecast.py
```

Note that these are all live plots so it is not reproducable.

#### Bitcoin features for LightGBM Model:

##### 1. Price Changes
- `P_1h`: 1-hour return (% change)
- `P_7h`: 7-hour return (% change)

##### 2. Volume Changes
- `V_1h`: 1-hour volume change (% change)
- `V_7h`: 7-hour volume change (% change)

##### 3. Price-Volume Interaction
- `PV_1h`: P_1h × V_1h
- `PV_7h`: P_7h × V_7h

##### 4. Volatility & Liquidity
- `vol`: rolling standard deviation of 1-hour returns (7-period window)
- `ratio`: Volume / Close (liquidity proxy)

##### 5. Lagged Price
- `Close_lag1`: previous period’s close price

##### 6. Factor-Based Features (added later from Fama-MachBeth)
- `CFO` only: Cumulative Forecast Oscillator (deviation from linear trend)
(Other factors not included due to low significance)

These features are combined and cleaned before being fed into LightGBM for short-term Bitcoin price forecasting.

#### Bitcoin Price Visualization Actual Vs. Predicted

The visualization for bitcoin price prediction over the past month is shown below, where last point shows the next hour price.

Average price range: ~$112,400 ; 

RMSE: 424.12 → error ~0.37% ; 

MAE: 282.39 → error ~0.25%

<img width="3561" height="1739" alt="bitcoin_prediction" src="https://github.com/user-attachments/assets/df5d1498-d890-428f-b7de-a18df0151735" />

Below shows the real bitcoin price at the next hour online.


<img width="753" height="489" alt="Screenshot 2025-09-18 at 16 05 52" src="https://github.com/user-attachments/assets/4a2d8ceb-8e50-4f6e-aaf0-b42eb307a0cf" />



### 2. Next 4 hour bitcoin price

Similary, with the same enginerring features, the model also trains for next 4 hour prediction. The visualization for bitcoin price prediction over the past month is shown below with 4 hour interval, where last point shows the next 4 hour price. 

Comparing with next-hour prediction, a 4-hour prediction is less noisy therefore has better visual effects. However, the model behaves slightly worse due to higher RMSE and MAE.

Average price range: ~$112,400 ; 

RMSE: 555.36 → error ~0.49% ; 

MAE: 415.88 → error ~0.37%

<img width="3560" height="1754" alt="bitcoin_prediction_4h" src="https://github.com/user-attachments/assets/b2bbb6c0-fb11-4ffc-a13d-5605abbfe270" />

### 3. TC, PWMA, and CFO factors

#### TC (Trend Confidence)
**Definition:**  
TC measures how well past price data can be explained by a linear trend over a rolling window. A higher TC indicates stronger trend predictability.

**Formula:**


$P_t$ = $\alpha$ + $\beta$ t + $\epsilon_t$

$$
TC = R^2 = 1 - \frac{\sum_{i=1}^{n} (\hat{P}_i - P_i)^2}{\sum_{i=1}^{n} (P_i - \bar{P})^2}
$$

Where:  
- $P_i$ = actual price  
- $\hat{P}_i$ = predicted price from linear regression  
- $\bar{P}$ = mean price in the window  

---

#### PWMA (Pascal-Weighted Moving Average)

**Definition:**  
PWMA is a weighted moving average where weights are derived from **Pascal’s triangle**. It smooths price series while preserving trend information.

**Formula:**

$$
w_k = \frac{\binom{n-1}{k}}{\sum_{j=0}^{n-1} \binom{n-1}{j}}, \quad k = 0,1,...,n-1
$$

$$
PWMA_t = \sum_{k=0}^{n-1} w_k \cdot P_{t-n+1+k}
$$

Where:  
- $P_{t-n+1+k}$ = price at position $k$ in the window  
- $\binom{n-1}{k}$ = binomial coefficient from Pascal's triangle  

---

#### CFO (Cumulative Forecast Oscillator)

**Definition:**  
CFO measures the deviation of the current price from a short-term linear prediction. It is scaled as a percentage of the current price and acts as a momentum signal.

**Formula:**

$$
\hat{P}_t = \text{LinearRegression}(P_{t-n+1:t})
$$

$$
CFO_t = 100 \times \frac{P_t - \hat{P}_t}{P_t}
$$

Where:  
- Positive CFO → price above predicted trend  
- Negative CFO → price below predicted trend  

Below shows the factor values for just bitcoin:


<img width="811" height="308" alt="Screenshot 2025-09-18 at 17 03 50" src="https://github.com/user-attachments/assets/32d8cb92-45ae-4766-9508-1f82e616f20a" />



### 4. Fama-MachBeth Estimation Across Cryptos

Two-step regression method to estimate factor risk premia in panel data.

**Cross-sectional regression per time period:**

$$
     R_{i,t} = \gamma_{0,t} + \gamma_{1,t} \beta_{i,1} + \dots + \gamma_{K,t} \beta_{i,K} + \epsilon_{i,t}
$$

**Time-series averaging of factor premia:**

$$
     \hat{\gamma}_k = \frac{1}{T} \sum_{t=1}^T \gamma_{k,t}
$$

**Purpose:** Provides robust estimates of factor effects on asset returns while accounting for cross-sectional dependence.


With the definitions of the above factors, the Fama-MachBeth Estimation is run across these assests: Bitcoin, Ethereum, Solana, Binance Coin, Cardano, Ripple, Dogecoin, Polkadot, Avalanche, Polygon, Litecoin, Chainlink, Cosmos, NEAR Protocol, Fantom, Algorand, TRON, Stellar, Shiba Inu, Internet Computer

<img width="681" height="445" alt="Screenshot 2025-09-18 at 16 12 28" src="https://github.com/user-attachments/assets/24f2de40-981e-454e-9c41-9bf573f8bef9" />

The results show that the model is 49.85% effective in describing the crypto trends from R-squared. Parameter estimates for all factors show that TC is the most significant with P-value = 0; PWMA is slightly correlated with P-value = 0.359.




