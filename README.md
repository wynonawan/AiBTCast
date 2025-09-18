# AiCryptoCast
AiBTCast is a Python project that predicts the real-time next-hour Bitcoin (BTC-USD) price using LightGBM and various engineered features: price, volume, and CFO (Cumulative Forecast Oscillator). Such engineering also performs well for next-4-hour prediction, which will be used in comparison in this repo.
In addition, this model also includes factor extraction for TC (Trend Confidence), PWMA (Predictive Weighted Moving Average), incorporating the FamaMacBeth estimation for all factors across multiple crypto assets. THe model reaches decent accuracy with ~0.25% overall error calculated from MAE, and ~0.34% from RMSE; it successfully captures the future movement trend.
This project is built for personal interest

## Contents
- [Install]()
- Bitcoin Forcast:
  - [Next-hour bitcoin price]()
  - [Next-4hour bitcoin price]()
  - [TC, PWMA, CFO factors]()
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





 paper: [Cryptocurrency price forecasting](https://www.sciencedirect.com/science/article/pii/S1057521923005719#:~:text=.%2C%202022). 

