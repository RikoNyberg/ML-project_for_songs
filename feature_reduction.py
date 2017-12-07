import find_correlation
import pandas as pd


def run(TRAINING_DATA):
    """
    Given a path to the training data CSV file, this will find highly correlated features,
    and return a list of only relevant features as index of the columns,

    Parameters
    ----------
    TRAINING_DATA : a path to the training data CSV file

    Returns
    -------
    list : the indexes of useful feature columns to use in the classification
    """

    print('Feature reduction:\nChecking if there is any correlation between the features in the training data...')
    df = pd.read_csv(TRAINING_DATA, header=None)
    columns_to_take_away, count_of_columns = find_correlation.run(df)

    if len(columns_to_take_away) > 0:
      columns = list(range(count_of_columns))
      for i in sorted(columns_to_take_away, reverse=True):
          del columns[i]
    else:
      columns = []

    return columns