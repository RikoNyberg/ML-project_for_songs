import import_data
import k_fold
import log_loss_regression_for_data

import numpy as np
import tensorflow as tf

# Data sets
TRAINING_DATA = "data/train_data.csv"
TRAINING_LABELS = "data/train_labels.csv"
TRAINING_DATA_labelled = 'data/train_data_labeled.csv'
TEST_DATA = "data/test_data.csv"

def main():
  # Load datasets as Bunch.
  
  # Two ways of downloading the training_set:
  # 1. With tensorflow function:
  training_set = tf.contrib.learn.datasets.base.load_csv_with_header(
      filename=TRAINING_DATA_labelled,
      target_dtype=np.int,
      features_dtype=np.float32)
  # 2. Self made function without the labels in the same file:
  # training_set = import_data.run(TRAINING_DATA, TRAINING_LABELS)
  
  test_set = import_data.run(TEST_DATA, 0)
  print('Data imported')

  print('Doing K-fold validation for the Logistic Regression model:')
  k_fold.run(training_set, test_set)

  print('Doing Logistic Regression for the training data and label prediction for the test data:')
  log_loss_regression_for_data.run(training_set, test_set)

  return

if __name__ == "__main__":
    main()
