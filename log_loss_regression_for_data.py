from sklearn import linear_model
import pandas as pd
import numpy as np

# TODO: Documentation on what is done

def run(train_bunch, test_bunch):

    X_train = train_bunch.data
    y_train = train_bunch.target
    X_test = test_bunch.data

    C = [1, 0.5]
    penalty = ['l2', 'l1']

    for C, penalty in zip(C, penalty):
        print('\n####################################')
        print('Logistic Regression (C={}, penalty={}) calculation starts...'.format(
            C, penalty))
        
        clf = linear_model.LogisticRegression(penalty=penalty, C=C)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        pred = clf.predict_proba(X_test)


        # Adding the count numbers to the label list
        y_pred = [list(range(1, len(y_pred) + 1)), y_pred]
        y_pred_transpose = np.transpose(np.array(y_pred))
        
        y_pred_transpose = pd.DataFrame(
            y_pred_transpose, columns=['Sample_id', 'Sample_label'])
        y_pred_transpose.to_csv('predictions/predicted_labels_C={},_penalty={}.csv'.format(C, penalty),
                    index=False, header=True, sep=',')

        pred = pd.DataFrame(pred, columns=[
                            'Class_1', 'Class_2', 'Class_3', 'Class_4', 'Class_5', 'Class_6', 'Class_7', 'Class_8', 'Class_9', 'Class_10'])
        pred.index = pred.index + 1
        pred.index.name = 'Sample_id'
        pred.to_csv('predictions/predicted_probabilities_C={},_penalty={}.csv'.format(C, penalty),
                    index=True, header=True, sep=',')

        print('predicted_labels_C={},_penalty{}.csv and predicted_probabilities_C={},_penalty{}.csv have been added to predictions-folder'.format(C, penalty, C, penalty))
        print('####################################\n')
    
    return
