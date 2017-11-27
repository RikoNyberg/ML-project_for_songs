import numpy as np
from sklearn import datasets

def load_data(X_path, y_path):
  """
  Loads training or testing data and returns it as a Dictionary with matrix of 
  training/testing values as 'data' and each rows labes in a list as 'target'
  
  Parameters
  ----------
  X_path : Path to training or testing data as txt file (with '\n' and ',' separators)
  y_path : Path to training or testing labels as txt file (with '\n' separator)

  Returns
  -------
  Bunch : Dictionary-like object that exposes its keys as attributes. Dictionary with matrix of training/testing values as 'data' 
         and each rows labes in a list as 'target'.
      More info: https://kite.com/docs/python;sklearn.datasets.base.Bunch
  """
  data_list = []
  labels_list = []
  bunch_of_data_and_labels = datasets.base.Bunch()

  text_file = open(X_path, 'r')
  lines = text_file.read().split('\n')
  del lines[-1]
  for line in lines:
    row_of_data = line.split(',')
    int_row_of_data =[]
    for data in row_of_data:
      int_row_of_data.append(float(data))
    data_list.append(int_row_of_data)

  if y_path != 0:
    text_file = open(y_path, 'r')
    lines = text_file.read().split('\n')
    del lines[-1]
    for value in lines:
      labels_list.append(int(value))

  bunch_of_data_and_labels.data = np.array(data_list)
  bunch_of_data_and_labels.target = np.array(labels_list)

  return bunch_of_data_and_labels


def run(X_path, y_path):
  bunch_data_set = load_data(X_path, y_path)
  return bunch_data_set
