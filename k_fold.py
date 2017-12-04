from sklearn import linear_model
import k_fold_data_log_loss
from sklearn.model_selection import KFold
import csv

# TODO: Documentation on what is done


def run(train_bunch, test_bunch):
    print('Running K-fold...')
    X = train_bunch.data
    y = train_bunch.target
    kf = KFold(n_splits=5, shuffle=True)
    kf.get_n_splits(X)

    csv_file = [['Training or Validation data', 'Accuracity',
                 'Sample size', 'Wrong labels', 'Used samples (test set)']]
    best_accuracy = 0
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        clf = linear_model.LogisticRegression()
        #clf = linear_model.LinearRegression()
        clf.fit(X_train, y_train)

        y_pred = clf.predict(X_test)
        wrong_labels = len(y_test) - \
            len([i for i, j in zip(y_test, y_pred) if i == j])
        test_accuracity = (len(y_test) - wrong_labels) / len(y_test)
        print('Validation data: {} wrong predicts out of {} samples. Accuracity: {}'.format(
            wrong_labels, len(y_test), test_accuracity))
        csv_file.append(['VALIDATION', test_accuracity,
                         len(y_test), wrong_labels, list(test_index)])

        y_pred = clf.predict(X_train)
        wrong_labels = len(y_train) - \
            len([i for i, j in zip(y_train, y_pred) if i == j])
        accuracity = (len(y_train) - wrong_labels) / len(y_train)
        print('Training data: {} wrong predicts out of {} samples. Accuracity: {}'.format(
            wrong_labels, len(y_train), accuracity))
        csv_file.append(['TRAIN', accuracity, len(
            y_train), wrong_labels, list(train_index)])

        if best_accuracy < test_accuracity:
            best_accuracy = test_accuracity
            X_best_train, X_best_test = X[train_index], X[test_index]
            y_best_train, y_best_test = y[train_index], y[test_index]

    # Save results to k_fold_validation.csv file
    with open("predictions/k_fold_validation.csv", "w") as f:
        writer = csv.writer(f)
        writer.writerows(csv_file)
    print('Results saved to k_fold_validation.csv file')

    # Doing the classification with the best k-fold data
    print('Best test accuracity if {} with train dataset of {} datapoint'.format(
        best_accuracy, len(y_best_train)))

    print('Calculating the classifier with the best training set...')
    clf = linear_model.LogisticRegression()
    clf.fit(X_best_train, y_best_train)

    k_fold_data_log_loss.run(clf, test_bunch)

    return
