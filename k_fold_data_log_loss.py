from sklearn import linear_model
import csv

import numpy as np

# TODO: Documentation on what is done

def run(clf, test_bunch):
  print('Logistic Regression calculation starts...')

  X_test = test_bunch.data
  y_pred = clf.predict(X_test) 
  pred = clf.predict_proba(X_test)
  
  print('####################################')
  # Adding the count numbers to the label list
  y_pred = [list(range(1, len(y_pred) + 1)), y_pred]
  y_pred_transpose = np.transpose(np.array(y_pred))

  with open("predictions/k_fold_predicted_labels.csv", "w") as f:
      writer = csv.writer(f)
      writer.writerows(y_pred_transpose)
  
  with open("predictions/k_fold_predicted_probabilities.csv", "w") as f:
    writer = csv.writer(f)
    writer.writerows(pred)
  
  print('k_fold_predicted_labels.csv and k_fold_predicted_probabilities.csv have been added to predictions-folder')
  print('####################################')

  return 
