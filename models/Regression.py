import logging

import numpy as np

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.svm import LinearSVR


def regression(x_train, x_test, y_train, y_test, nr_classes, epsilon=None,
               c=None):
    """Regression model.

    x_train : training data for the training set.
    x_test : training data for the test set.
    y_train : labels for the training set (x_train).
    y_test : labels for the test set (y_train).
    nr_classes : the amount of classes possible.
    epsilon :
    c : regularisation parameter
    """
    logging.debug("> Running linear support vector regression")
    if not epsilon or not c:
        # If we are missing either epsilon or c (we assume they come as a
        # pair) we use grid search cross validation to find the best values for
        # them.
        logging.debug("> Finding best value for epsilon and c...")
        params = [{"epsilon": np.logspace(-10, -1, 5)},
                  {"C": np.logspace(-4, 2, 6)}]
        svm_model = GridSearchCV(LinearSVR(), param_grid=params, n_jobs=-1,
                                 cv=8)
        svm_model.fit(x_train, y_train)
        epsilon = svm_model.best_estimator_.epsilon
        c = svm_model.best_estimator_.C
        logging.debug("> Found best epsilon value at " + str(epsilon))
        logging.debug("> Found best C value at " + str(c))

    svm_model = LinearSVR(epsilon=epsilon, C=c).fit(x_train, y_train)
    svm_predictions = svm_model.predict(x_test)

    # As we are using regression we need to round the predicted values to
    # a class (integer, amount of stars) and clamp the value between 0 and
    # maximum amount of stars.
    rounded_predictions = np.round(svm_predictions)
    rounded_predictions = [min(nr_classes - 1, max(0, x))
                           for x in rounded_predictions]

    accuracy = accuracy_score(rounded_predictions, y_test)

    cm = confusion_matrix(y_test, rounded_predictions)
    logging.debug("Confusion matrix:\n{cm}".format(cm=cm))

    return accuracy, cm, svm_model.coef_