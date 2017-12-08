from sklearn import linear_model
import k_fold_data_log_loss
from sklearn.model_selection import KFold
import pandas as pd
from pandas_ml import ConfusionMatrix
import matplotlib.pyplot as plt
from sklearn import datasets

# TODO: Documentation on what is done


def save_k_fold_results_to_CSV(csv_file):
    # Save results to k_fold_validation.csv file
    csv_file = pd.DataFrame(
        csv_file, columns=['Training or Validation data', 'Accuracity', 'Sample size', 'Wrong labels', 'Used samples(test set)'])
    csv_file.to_csv('predictions/k_fold_validation.csv',
                            index=False, header=True, sep=',')
    print('\n####################################')
    print('K-fold results saved to predictions/k_fold_validation.csv file')
    print('####################################\n')

def plot_confusion_matrix(clf, X_best_test, y_best_test, matrix_name):
    # Doing the confusion matrix for the best K-validated training set
    y_best_pred = clf.predict(X_best_test)
    confusion_matrix = ConfusionMatrix(y_best_test, y_best_pred)
    #print("Confusion matrix:\n{}".format(confusion_matrix))
    confusion_matrix.plot(normalized=True)
    plt.savefig('confusion_matrixes/K-fold_matrix_{}.png'.format(matrix_name))
    print('Saved Confusion matrix of the previous test to confusion_matrixes/K-fold_matrix_{}.png\n'.format(matrix_name))

    return confusion_matrix

def run(train_bunch, test_bunch):
    n_splits = 5
    print('Running K-fold with {} splits...\n'.format(n_splits))
    X = train_bunch.data
    y = train_bunch.target
    kf = KFold(n_splits=n_splits, shuffle=True)
    kf.get_n_splits(X)

    csv_file = []
    best_accuracy = 0
    matrix_name = 1
    for train_index, test_index in kf.split(X):
        print('Split {}:'.format(matrix_name))
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        clf = linear_model.LogisticRegression(penalty='l1', C=0.5)
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

        plot_confusion_matrix(clf, X_test, y_test, matrix_name)
        matrix_name += 1 
        if best_accuracy < test_accuracity:
            clf_best = clf
            best_accuracy = test_accuracity
            X_best_train, X_best_test = X[train_index], X[test_index]
            y_best_train, y_best_test = y[train_index], y[test_index]

    save_k_fold_results_to_CSV(csv_file)

    # Doing the classification with the best k-fold data
    print('Best test accuracity if {} with train dataset of {} datapoint'.format(
        best_accuracy, len(y_best_train)))
    matrix_name = 'best_accuracity_({})'.format(best_accuracy)
    confusion_matrix = plot_confusion_matrix(
        clf_best, X_best_test, y_best_test, matrix_name)
    print("Confusion matrix for the best K-fold validated training set:\n{}".format(confusion_matrix))
    
    best_train_bunch = datasets.base.Bunch(
        data=X_best_train, target=y_best_train)
    
    k_fold_data_log_loss.run(best_train_bunch, test_bunch)
    return
