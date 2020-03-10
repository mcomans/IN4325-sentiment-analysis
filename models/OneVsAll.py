import logging

import numpy as np

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC


def classify_ova(x_train, x_test, y_train, y_test, c=None):
    """One versus all classifier.

    x_train : training data for the training set.
    x_test : training data for the test set.
    y_train : labels for the training set (x_train).
    y_test : labels for the test set (y_train).
    c : regularisation parameter
    """
    logging.debug("> Running One-vs-All classifier")
    if not c:
        # If c has not been set we estimate it using grid search cross
        # validation.
        logging.debug("> Finding best value C...")
        model_to_set = OneVsRestClassifier(SVC(kernel="linear"))
        params = {"estimator__C": np.logspace(-4, 2, 6)}

        svm_model = GridSearchCV(model_to_set, param_grid=params, n_jobs=-1,
                                 cv=8)
        svm_model.fit(x_train, y_train)
        c = svm_model.best_estimator_.estimator.C
        logging.debug("> Found best C value at " + str(c))

    # Given our value for C train the model.
    svm_model = OneVsRestClassifier(
        SVC(kernel="linear", C=c)).fit(x_train, y_train)

    svm_predictions = svm_model.predict(x_test)
    accuracy = svm_model.score(x_test, y_test)

    cm = confusion_matrix(y_test, svm_predictions)
    logging.debug("Confusion matrix:\n{cm}".format(cm=cm))

    ova_coef_list = []

    for estimator in svm_model.estimators_:
        ova_coef_list.append(estimator.coef_.toarray()[0])

    return accuracy, cm, ova_coef_list