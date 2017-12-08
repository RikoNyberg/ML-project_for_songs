import pandas as pd
import numpy as np

def run(df, thresh=0.8):
    """
    Given a numeric pd.DataFrame, this will find highly correlated features,
    and return a list of features to remove and the count of collums

    Parameters
    ----------
    df : pd.DataFrame
    thresh : correlation threshold, will remove one of pairs of features with
              a correlation greater than this value
    Returns
    -------
    set : features to remove
    int : count of collums
    """
    
    corrMatrix = df.corr()
    corrMatrix.loc[:, :] = np.tril(corrMatrix, k=-1)

    already_in = set()
    result = []

    for col in corrMatrix:
        perfect_corr = corrMatrix[col][corrMatrix[col] > thresh].index.tolist()
        if perfect_corr and col not in already_in:
            already_in.update(set(perfect_corr))
            perfect_corr.append(col)
            result.append(perfect_corr)

    select_nested = [f[1:] for f in result]
    select_flat = [i for j in select_nested for i in j]
    
    select_flat = set(select_flat)

    if len(select_flat) > 0:
      print('Correlations found between features. {} out of {} features will be removed...'.format(
          len(select_flat), len(corrMatrix)))
    else:
      print('No correlations found. All {} features are used in the classification...'.format(len(corrMatrix)))
    return select_flat, len(corrMatrix)
