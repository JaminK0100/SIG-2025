
import numpy as np

##### TODO #########################################
### IMPLEMENT 'getMyPosition' FUNCTION #############
### TO RUN, RUN 'eval.py' ##########################

nInst = 50
currentPos = np.zeros(nInst)

# Sample getMyPosition from SIG
def getMyPosition_example(prcSoFar):
    global currentPos
    (nins, nt) = prcSoFar.shape
    if (nt < 2):
        return np.zeros(nins)
    lastRet = np.log(prcSoFar[:, -1] / prcSoFar[:, -2])      
    lNorm = np.sqrt(lastRet.dot(lastRet))
    lastRet /= lNorm
    rpos = np.array([int(x) for x in 5000 * lastRet / prcSoFar[:, -1]])
    currentPos = np.array([int(x) for x in currentPos+rpos])
    return currentPos
# Problems: the function is too simple, right now it is simply saying long on whatever went up yesterday and short on what went down
# Need to make new strategy



"""Smoothed Inverse Volatility"""
def getMyPosition_InverseVolatility_SmoothedTrends(prcSoFar): #Jamin
    global currentPos
    (nInst, nt) = prcSoFar.shape
    
    # No analysis for instances < 2
    if nt < 20:
        return np.zeros(nInst)

    # Moving average crossover frame period (Can be changed later)
    short_window = 5
    long_window = 20

    short_ma = np.mean(prcSoFar[:, -short_window:], axis=1)
    long_ma = np.mean(prcSoFar[:, -long_window:], axis=1)

    # Signal = momentum (go long if short_ma > long_ma, short if opposite)
    raw_signal = short_ma - long_ma

    # Normalise and scale by inverse volatility in range of sample
    # Use rolling standard deviation for volatility (can be changed if something works better)
    volatility = np.std(prcSoFar[:, -long_window:], axis=1) + 1e-6  # Avoid div by 0
    signal = raw_signal / volatility

    # Scale positions 
    dollar_target = 4000  # target $ exposure per signal (also temprorary, might change this)
    # Use 2000 for lower turnover values and 6000 for more aggression
    current_prices = prcSoFar[:, -1]
    pos = dollar_target * signal / current_prices
    pos = np.round(pos).astype(int)

    currentPos = pos
    return currentPos


"""Anti-Martingales"""
# Store recent PLs to assess performance better in function
rollingPL = []
def getMyPosition_AntiMartingales(prcSoFar): #Jamin
    global currentPos, rollingPL
    (nInst, nt) = prcSoFar.shape

    # No analysis < 21
    if nt < 21:
        return np.zeros(nInst)

    # Similar to above - Signal = Smoothed Momentum
    short_window = 5
    long_window = 20
    short_ma = np.mean(prcSoFar[:, -short_window:], axis=1)
    long_ma = np.mean(prcSoFar[:, -long_window:], axis=1)
    raw_signal = short_ma - long_ma

    # Volatility Scaling again with inversed volatility
    vol = np.std(prcSoFar[:, -long_window:], axis=1) + 1e-6
    signal = raw_signal / vol

    # Estimate yesterday's PnL
    yday_price = prcSoFar[:, -2]
    today_price = prcSoFar[:, -1]
    approx_pnl = np.sum((today_price - yday_price) * currentPos)
    rollingPL.append(approx_pnl)
    if len(rollingPL) > 10:
        rollingPL = rollingPL[-10:]

    # Anti-Martingale Adjustment part, using 4000 again as base (will need to research or tune this for sure)
    base_target = 4000
    if len(rollingPL) >= 5:
        mu = np.mean(rollingPL)
        sigma = np.std(rollingPL) + 1e-6
        sharpe = mu / sigma

        if sharpe > 1.0:
            dollar_target = 6000  # do well = risk a bit more
        elif sharpe < -0.5:
            dollar_target = 2000  # drawdown = reduce risk
        else:
            dollar_target = base_target
    else:
        dollar_target = base_target

    # Position Sizing similar to other function
    prices = prcSoFar[:, -1]
    pos = dollar_target * signal / prices
    pos = np.round(pos).astype(int)

    currentPos = pos
    return currentPos


"""MACD with crossover and stop losses (With Anti-Martingales Adjustment)"""
prev_positions = np.zeros(nInst, dtype=int)
entry_prices = np.zeros(nInst)
rollingPL = []

def getMyPosition(prcSoFar):
    global prev_positions, entry_prices, rollingPL

    nInst, nt = prcSoFar.shape
    positions = np.zeros(nInst, dtype=int)

    # No analysis < 30
    if nt < 30:
        return positions  # Not enough data for MACD otherwise

    # Strategy parameters (can change as well)
    short_window = 12
    long_window = 26
    signal_window = 5
    stop_loss_amount = 500

    # Adjust max investment based on recent PL (anti-martingale)
    maxInvestAmt = update_dollar_target(prcSoFar)

    for inst in range(nInst):
        prices = prcSoFar[inst, :]
        current_price = prices[-1]

        # Get MACD components
        short_ema = calculate_ema(prices, short_window)
        long_ema = calculate_ema(prices, long_window)
        macd = short_ema - long_ema
        signal_line = calculate_ema(macd, signal_window)
        macd_hist = macd - signal_line

        # Position sizing
        position_size = int(maxInvestAmt / current_price)

        # Stop-loss logic opart
        if prev_positions[inst] != 0:
            entry_price = entry_prices[inst]
            pnl = prev_positions[inst] * (current_price - entry_price)
            if pnl < -stop_loss_amount:
                positions[inst] = 0
                prev_positions[inst] = 0
                entry_prices[inst] = 0
                continue

        # Entry signals
        if macd_hist[-1] > 0 and macd_hist[-2] <= 0:
            # Bullish crossover
            positions[inst] = position_size
            prev_positions[inst] = position_size
            entry_prices[inst] = current_price

        elif macd_hist[-1] < 0 and macd_hist[-2] >= 0:
            # Bearish crossover
            positions[inst] = -position_size
            prev_positions[inst] = -position_size
            entry_prices[inst] = current_price

        else:
            # Maintain current position
            positions[inst] = prev_positions[inst]

    return positions

# Support function
def calculate_ema(prices, window):
    """Standard exponential moving average (recursive)"""
    ema = np.zeros_like(prices)
    alpha = 2 / (window + 1)
    ema[0] = prices[0]
    for t in range(1, len(prices)):
        ema[t] = alpha * prices[t] + (1 - alpha) * ema[t-1]
    return ema

# Support Function
def update_dollar_target(prcSoFar):
    """Adjust position sizing based on recent PnL (anti-martingale)"""
    global prev_positions, rollingPL

    if prcSoFar.shape[1] < 2:
        return 5000  # default

    yday = prcSoFar[:, -2]
    today = prcSoFar[:, -1]
    approx_pnl = np.sum((today - yday) * prev_positions)
    rollingPL.append(approx_pnl)
    if len(rollingPL) > 10:
        rollingPL = rollingPL[-10:]

    if len(rollingPL) >= 5:
        mu = np.mean(rollingPL)
        sigma = np.std(rollingPL) + 1e-6
        sharpe = mu / sigma

        if sharpe > 1.0:
            return 7000
        elif sharpe < -0.5:
            return 3000
        else:
            return 5000
    else:
        return 5000
        # Again, all these numbers that are hard coded can be changed, might need to hypertune them somewhat

