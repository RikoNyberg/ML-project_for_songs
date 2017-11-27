from sklearn import linear_model
from sklearn.model_selection import KFold

# TODO: Documentation on what is done

def run(train_bunch):
  print('Running K-fold...')
  X = train_bunch.data
  y = train_bunch.target
  kf = KFold(n_splits=5, shuffle=True, random_state=2)
  kf.get_n_splits(X)

  csv_file = [['Training or Validation data', 'Accuracity',
               'Sample size', 'Wrong labels', 'Used samples (test set)']]
  for train_index, test_index in kf.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    clf = linear_model.LogisticRegression()
    #clf = linear_model.LinearRegression()
    clf.fit(X_train, y_train)
    
    y_pred = clf.predict(X_test) 
    wrong_labels = len(y_test) - \
        len([i for i, j in zip(y_test, y_pred) if i == j])
    accuracity = (len(y_test) - wrong_labels) / len(y_test)
    print('Validation data: {} wrong predicts out of {} samples. Accuracity: {}'.format(
        wrong_labels, len(y_test), accuracity))
    csv_file.append(['VALIDATION', accuracity, len(y_test), wrong_labels, test_index])

    y_pred = clf.predict(X_train)
    wrong_labels = len(y_train) - \
        len([i for i, j in zip(y_train, y_pred) if i == j])
    accuracity = (len(y_train) - wrong_labels) / len(y_train)
    print('Training data: {} wrong predicts out of {} samples. Accuracity: {}'.format(
        wrong_labels, len(y_train), accuracity))
    csv_file.append(['TRAIN', accuracity, len(
        y_train), wrong_labels, train_index])
  
  # Save results to k_fold_validation.csv file
  with open("k_fold_validation.csv", "w") as f:
      writer = csv.writer(f)
      writer.writerows(csv_file)

  print('Results saved to k_fold_validation.csv file')
  return
