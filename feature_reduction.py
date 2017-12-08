import find_correlation
import pandas as pd

def run(TRAINING_DATA):
    """
    Given a path to the training data CSV file, this will find highly correlated features
    and return a list of only relevant features as index of the columns,

    Parameters
    ----------
    TRAINING_DATA : a path to the training data CSV file

    Returns
    -------
    list : indexes of useful feature columns to use in the classification
    """

    print('Feature reduction:\nChecking if there is any correlation between the features in the training data...')
    df = pd.read_csv(TRAINING_DATA, header=None)

    columns_to_take_away, count_of_orig_columns = find_correlation.run(df)
    columns = set(range(count_of_orig_columns))

    columns = columns - columns_to_take_away

    return columns, count_of_orig_columns
