import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

##### TODO #########################################
### IMPLEMENT 'getMyPosition' FUNCTION #############
### TO RUN, RUN 'eval.py' ##########################

nInst = 50
currentPos = np.zeros(nInst)

def loadPrices(fn):
    global nt, nInst
    df=pd.read_csv(fn, sep='\s+', header=None, index_col=None)
    (nt,nInst) = df.shape
    return (df.values).T



def getMyPosition(prcSoFar):
    return

# prob just do correlation matrix and find highest correlations
# need a bench mark correlation tho
def find_pairs(prices, threshold):
    """
    return all pairs that are highly correlated
    """
    corr_matrix = np.corrcoef(prices)
    n = corr_matrix.shape[0]
    high_corr_pairs = []

    for i in range(n):
        for j in range(i + 1, n):
            if corr_matrix[i, j] >= threshold:
                high_corr_pairs.append((i, j, corr_matrix[i, j]))
                
    print(f"Found {len(high_corr_pairs)} highly correlated pairs (threshold={threshold}):")
    for i, j, corr in high_corr_pairs:
        print(f"Instrument {i} and {j} → Corr = {corr:.3f}")
    return high_corr_pairs

if __name__ == "__main__":
    pricesFile="prices.txt"
    prcAll = loadPrices(pricesFile)
    find_pairs(prcAll, 0.8)
    print ("Loaded %d instruments for %d days" % (nInst, nt))