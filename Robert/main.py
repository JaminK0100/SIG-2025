import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import statsmodels.api as sm
import statsmodels.tsa.stattools as ts
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import acf
import pandas_ta as ta

##### TODO #########################################
### IMPLEMENT 'getMyPosition' FUNCTION #############
### TO RUN, RUN 'eval.py' ##########################


class Algorithm:
    """
    For all 3 strats - we can prob run a grid search to find optimal parameters
    """

    def __init__(self, filename):
        self.filename = filename
        self.nInst = 0
        self.nt = 0
        self.window = 30
        self.currentPos = None
        self.highly_corr_pairs = None
        self.coint_pairs = None
        self.stationary_pairs = None
        self.unused_instruments = None
        self.stationary_instruments = None
        self.paired_instruments = None
        self.prices = None

    def loadPrices(self):
        df=pd.read_csv(self.filename, sep='\s+', header=None, index_col=None)
        (nt,nInst) = df.shape
        self.nt = nt
        self.nInst = nInst
        self.currentPos = np.zeros(self.nInst)
        self.prices = (df.values).T

    # prob just do correlation matrix and find highest correlations
    # need a bench mark correlation tho
    def find_pairs(self, threshold):
        """
        return all pairs that are highly correlated
        """
        corr_matrix = np.corrcoef(self.prices)
        n = corr_matrix.shape[0]
        high_corr_pairs = []

        for i in range(n):
            for j in range(i + 1, n):
                if corr_matrix[i, j] >= threshold:
                    high_corr_pairs.append((i, j, corr_matrix[i, j]))
                    
        print(f"Found {len(high_corr_pairs)} highly correlated pairs (threshold={threshold}):")
        for i, j, corr in high_corr_pairs:
            print(f"Instrument {i} and {j} -> Corr = {corr:.3f}")
        self.highly_corr_pairs = high_corr_pairs

    def test_coint(self):
        coint_pairs = []
        for i, j, corr in self.highly_corr_pairs:
            _, p_value, _ = ts.coint(self.prices[i], self.prices[j])
            if p_value < 0.05:
                print(f"Instruments {i} and {j} are cointegrated (p-value = {p_value:.4f})")
                coint_pairs.append((i, j, corr))
        self.coint_pairs = coint_pairs

    def test_spread_stationarity(self):
        stationary_pairs = []

        for i, j, corr in self.coint_pairs:
            y = self.prices[i]
            x = self.prices[j]

            # OLS regression: y = alpha + beta * x
            x_with_const = sm.add_constant(x)
            model = sm.OLS(y, x_with_const).fit()
            alpha, beta = model.params

            # Compute spread
            spread = y - (alpha + beta * x)

            # ADF test
            result = adfuller(spread)
            p_value = result[1]

            if p_value < 0.05:
                print(f"Pair ({i}, {j}) has stationary spread (ADF p = {p_value:.4f})")
                stationary_pairs.append((i, j, corr))

        print(f"\nFound {len(stationary_pairs)} stationary pairs out of {len(self.coint_pairs)}")
        self.stationary_pairs = stationary_pairs

    #TODO: grid search to optimise parameters for arbitrage/ pairs trading strat
    def grid_search(self):
        pass
            
    def get_unused_instruments(self):
        used = set()
        for i, j, corr in self.stationary_pairs:
            used.add(i)
            used.add(j)
        n = self.prices.shape[0]
        print(n)
        unused = []
        for i in range(n):
            if i not in used:
                unused.append(i)
        return unused
        
    def test_stationary_instruments(self):
        unused_instruments = self.get_unused_instruments()
        stationary = []
        for i in unused_instruments:
            result = adfuller(self.prices[i])
            p_value = result[1]
            if p_value < 0.05:
                print(f"Instrument {i} ADF p-value: {p_value:.4f}")
                stationary.append(i)
        self.stationary_instruments = stationary
        
    def set_paired_instruments(self):
        result = set()
        for i, j, corr in self.stationary_pairs:
            result.add(i)
            result.add(j)
        paired_instruments = list(result)
        self.paired_instruments = paired_instruments
        
    def get_not_stationary_not_pairs_insts(self):
        not_stationary_not_pairs = []
        for i in range(self.nInst):
            if i not in self.paired_instruments and i not in self.stationary_instruments:
                not_stationary_not_pairs.append(i)
        return not_stationary_not_pairs
    
    def autocorr_lag1(self, prices):
        returns = np.diff(prices) / prices[:-1]  # daily returns
        if len(returns) < 2:
            return 0
        autocorr_values = acf(returns, nlags=1, fft=False)
        return autocorr_values[1]  # lag-1 autocorrelation
    
    def get_trend_instruments(self):
        not_stationary_not_pairs = self.get_not_stationary_not_pairs_insts()
        trend_instruments = []
        for i in not_stationary_not_pairs:
            prices = self.prices[i]
            autocorr = self.autocorr_lag1(prices)
            if autocorr > 0:
                trend_instruments.append(i)
        return trend_instruments


def trend_following(prcSoFar, long_window, short_window, adx_window=14, adx_threshold=30):
    trend_instruments = algo.get_trend_instruments()
    positions = np.zeros(prcSoFar.shape[0])
    
    for i in trend_instruments:
        prices = prcSoFar[i]
        if len(prices) < max(long_window, adx_window):
            continue
        
        price_series = pd.Series(prices)
        
        # Calculate short and long moving averages
        short_ma = price_series.rolling(window=short_window).mean()
        long_ma = price_series.rolling(window=long_window).mean()
        
        # Calculate ADX (requires high, low, close - approximate with close here for simplicity)
        # Use close for all three
        df = pd.DataFrame({'close': prices})
        df['high'] = prices
        df['low'] = prices
        
        adx = ta.adx(df['high'], df['low'], df['close'], length=adx_window)
        curr_adx = adx['ADX_14'].iloc[-1]  # use ADX column with window 14
        
        curr_short_ma = short_ma.iloc[-1]
        curr_long_ma = long_ma.iloc[-1]

        if np.isnan(curr_short_ma) or np.isnan(curr_long_ma) or np.isnan(curr_adx):
            continue
        
        max_pos = int(10000 // prices[-1])
        
        # Only trade if ADX indicates strong trend
        if curr_adx > adx_threshold:
            if curr_short_ma > curr_long_ma:
                positions[i] = max_pos
            elif curr_short_ma < curr_long_ma:
                positions[i] = -max_pos
            else:
                positions[i] = 0
        else:
            # ADX too low = no strong trend, close position
            positions[i] = 0
            
    return positions

#TODO: implement just a mean rev strategy by itself to see how well it does
def mean_reversion(stationary_instruments, prcSoFar, window, threshold):
    positions = np.zeros(prcSoFar.shape[0])
    for i in stationary_instruments:
        prices = prcSoFar[i]
        if len(prices) < window:
            continue
        rolling_mean = pd.Series(prices).rolling(window=window).mean()
        rolling_std = pd.Series(prices).rolling(window=window).std()
        if rolling_std.iloc[-1] == 0:
            continue
        z_score = (prices[-1] - rolling_mean.iloc[-1]) / rolling_std.iloc[-1]
        
        max_pos = int(10000 // prices[-1])
        
        if z_score > threshold:
            # Price is above mean by threshold - short
            positions[i] = -max_pos
        elif z_score < -threshold:
            # Price below mean - long
            positions[i] = max_pos
        else:
            positions[i] = 0  # close position
    return positions

#TODO: start implementing getMyPosition
def getMyPosition(prcSoFar):
    (nins, nt) = prcSoFar.shape
    if nt < algo.window:
        return algo.currentPos
    
    for i, j, corr in algo.stationary_pairs:
        inst1 = prcSoFar[i]
        inst2 = prcSoFar[j]
        price_i = prcSoFar[i, -1]
        price_j = prcSoFar[j, -1]
        
        inst2_const = sm.add_constant(inst2)
        model = sm.OLS(inst1, inst2_const).fit()
        alpha, beta = model.params

        # Calculate spread and z-score
        spread = inst1 - (alpha + beta * inst2)
        mean_spread = np.mean(spread)
        std_spread = np.std(spread)

        if std_spread == 0:
            continue  # Avoid division by zero

        z = (spread[-1] - mean_spread) / std_spread

        max_pos_inst1 = int(10000 // price_i)
        max_pos_inst2 = int(10000 // price_j)
        amount = min(max_pos_inst1, max_pos_inst2)
        if z > 1:
            algo.currentPos[i] -= amount
            algo.currentPos[j] += int(beta * amount)
        elif z < -1:
            algo.currentPos[i] += amount
            algo.currentPos[j] -= int(beta * amount)
        else:
            algo.currentPos[i] = 0
            algo.currentPos[j] = 0
    
    mean_rev_pos = mean_reversion(algo.stationary_instruments, prcSoFar, 60, 2.5)
    for i in algo.stationary_instruments:
        algo.currentPos[i] = mean_rev_pos[i]
        
    trend_following_pos = trend_following(prcSoFar, 60, 30)
    trend_instruments = algo.get_trend_instruments()
    for i in trend_instruments:
        algo.currentPos[i] = trend_following_pos[i]
    return algo.currentPos


pricesFile = "prices.txt"
algo = Algorithm(pricesFile)
algo.loadPrices()
algo.find_pairs(0.8)
algo.test_coint()
algo.test_spread_stationarity()
algo.set_paired_instruments()
algo.test_stationary_instruments()
res = algo.get_trend_instruments()
print(res)
print(len(res))

print ("Loaded %d instruments for %d days" % (algo.nInst, algo.nt))