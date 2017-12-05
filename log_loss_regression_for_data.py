from sklearn import linear_model
import pandas as pd
import numpy as np

# TODO: Documentation on what is done


def run(train_bunch, test_bunch):
    print('Logistic Regression calculation starts...')

    X_train = train_bunch.data
    y_train = train_bunch.target
    X_test = test_bunch.data

    clf = linear_model.LogisticRegression()
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    pred = clf.predict_proba(X_test)

    print('####################################')
    # Adding the count numbers to the label list
    y_pred = [list(range(1, len(y_pred) + 1)), y_pred]
    y_pred_transpose = np.transpose(np.array(y_pred))
    
    y_pred_transpose = pd.DataFrame(
        y_pred_transpose, columns=['Sample_id', 'Sample_label'])
    y_pred_transpose.to_csv('predictions/predicted_labels.csv',
                  index=False, header=True, sep=',')

    pred = pd.DataFrame(pred, columns=[
                          'Class_1', 'Class_2', 'Class_3', 'Class_4', 'Class_5', 'Class_6', 'Class_7', 'Class_8', 'Class_9', 'Class_10'])
    pred.index = pred.index + 1
    pred.index.name = 'Sample_id'
    pred.to_csv('predictions/predicted_probabilities.csv',
                index=True, header=True, sep=',')

    print('predicted_labels.csv and predicted_probabilities.csv have been added to predictions-folder')
    print('####################################')

    return
