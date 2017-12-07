import import_data
import k_fold
import log_loss_regression_for_data
import feature_reduction

import numpy as np
import pandas as pd
#import tensorflow as tf

# Data sets
TRAINING_DATA = "data/train_data.csv"
TRAINING_LABELS = "data/train_labels.csv"
#TRAINING_DATA_labelled = 'data/train_data_labeled.csv'
TEST_DATA = "data/test_data.csv"


def create_new_csv_with_useful_features(useful_features, orig_csv_filepath, new_csv_filepath):
    data = pd.read_csv(orig_csv_filepath, header=None)
    data.to_csv(new_csv_filepath,
                columns=useful_features, index=False, header=False)
    

def main():
    train_data_csv = TRAINING_DATA
    test_data_csv = TEST_DATA

    # Feature reduction
    useful_features = feature_reduction.run(TRAINING_DATA)
    
    if len(useful_features) > 0:
        TRAIN_DATA_REDUCED = "data/train_data_reduced_features.csv"
        create_new_csv_with_useful_features(
            useful_features, TRAINING_DATA, TRAIN_DATA_REDUCED)
        train_data_csv = TRAIN_DATA_REDUCED
        
        TEST_DATA_REDUCED = "data/test_data_reduced_features.csv"
        create_new_csv_with_useful_features(
            useful_features, TEST_DATA, TEST_DATA_REDUCED)
        test_data_csv = TEST_DATA_REDUCED
    
    # Load datasets as Bunch.
    # Two ways of downloading the training_set:
    # 1. With tensorflow function:
    # training_set = tf.contrib.learn.datasets.base.load_csv_with_header(
    #     filename=TRAINING_DATA_labelled,
    #     target_dtype=np.int,
    #     features_dtype=np.float32)
    # 2. Self made function without the labels in the same file:
    training_set = import_data.run(train_data_csv, TRAINING_LABELS)

    test_set = import_data.run(test_data_csv, 0)
    print('Data imported')

    print('Doing K-fold validation for the Logistic Regression model:')
    k_fold.run(training_set, test_set)

    print('Doing Logistic Regression for the training data and label prediction for the test data:')
    log_loss_regression_for_data.run(training_set, test_set)

    return


if __name__ == "__main__":
    main()
