from sklearn import linear_model
import csv

# TODO: Documentation on what is done

def run(train_bunch, test_bunch):
  print('Logistic Regression calculation starts...')

  X_train = train_bunch.data
  y_train = train_bunch.target
  X_test = test_bunch.data

  clf = linear_model.LogisticRegression()
  clf.fit(X_train, y_train)
  y_pred = clf.predict(X_test) 
  log_pred = clf.predict_log_proba(X_test)
  pred = clf.predict_proba(X_test)
  
  print('####################################')
  with open("predictions/predicted_labels.csv", "w") as f:
      writer = csv.writer(f)
      writer.writerow(y_pred)
  
  with open("predictions/predicted_probabilities.csv", "w") as f:
    writer = csv.writer(f)
    writer.writerows(pred)
  
  print('predicted_labels.csv and predicted_probabilities.csv have been added to predictions-folder')
  print('####################################')

  return 
