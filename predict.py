import numpy as np

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
from sklearn.svm import LinearSVR

import argparse

from configurations import configurations
from plot import plot_coef
import data

parser = argparse.ArgumentParser()
parser.add_argument("--debug", action="store_true", help="Enable debug mode")
parser.add_argument("-f", "--feature-importance", action="store_true",
                    help="Generate feature importance plot")
parser.add_argument("-c", "--configuration", choices=configurations,
                    default='replicate-pang', help="Configuration preset")
parser.add_argument("-o", "--output", help="Output filename",
                    default="results.csv")

args = parser.parse_args()


def __debug_log(text):
    """Print if --debug flag is set."""
    if args.debug:
        print(text)


def __read_labels(labels_filepath):
    """Read labels from file as integers."""
    with open(labels_filepath) as f:
        return [int(line) for line in f.readlines()]


def classify_ova(x_train, x_test, y_train, y_test, c=None):
    """One versus all classifier.

    x_train : training data for the training set.
    x_test : training data for the test set.
    y_train : labels for the training set (x_train).
    y_test : labels for the test set (y_train).
    c : regularisation parameter
    """
    __debug_log("> Running One-vs-All classifier")
    if not c:
        # If c has not been set we estimate it using grid search cross
        # validation.
        __debug_log("> Finding best value C...")
        model_to_set = OneVsRestClassifier(SVC(kernel="linear"))
        params = {"estimator__C": np.logspace(-4, 2, 6)}

        svm_model = GridSearchCV(model_to_set, param_grid=params, n_jobs=-1,
                                 cv=8)
        svm_model.fit(x_train, y_train)
        c = svm_model.best_estimator_.estimator.C
        __debug_log("> Found best C value at " + str(c))

    # Given our value for C train the model.
    svm_model = OneVsRestClassifier(
        SVC(kernel="linear", C=c)).fit(x_train, y_train)

    svm_predictions = svm_model.predict(x_test)
    accuracy = svm_model.score(x_test, y_test)

    cm = confusion_matrix(y_test, svm_predictions)
    __debug_log("Confusion matrix:\n{cm}".format(cm=cm))

    ova_coef_list = []

    for estimator in svm_model.estimators_:
        ova_coef_list.append(estimator.coef_.toarray()[0])

    return accuracy, cm, ova_coef_list


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
    __debug_log("> Running linear support vector regression")
    if not epsilon or not c:
        # If we are missing either epsilon or c (we assume they come as a
        # pair) we use grid search cross validation to find the best values for
        # them.
        __debug_log("> Finding best value for epsilon and c...")
        params = [{"epsilon": np.logspace(-10, -1, 5)},
                  {"C": np.logspace(-4, 2, 6)}]
        svm_model = GridSearchCV(LinearSVR(), param_grid=params, n_jobs=-1,
                                 cv=8)
        svm_model.fit(x_train, y_train)
        epsilon = svm_model.best_estimator_.epsilon
        c = svm_model.best_estimator_.C
        __debug_log("> Found best epsilon value at " + str(epsilon))
        __debug_log("> Found best C value at " + str(c))

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
    __debug_log("Confusion matrix:\n{cm}".format(cm=cm))

    return accuracy, cm, svm_model.coef_


def run(author, nr_classes, feature_vectors, feature_names, labels):
    """
    Run both the regression and ova models for a given author and amount
    of classes.

    author : the author to run the model for.
    nr_classes : the amount of classes possible (either 3 or 4).
    feature_vectors : the feature vectors.
    feature_names : the names of the features.
    labels : the correct label for each feature.
    """
    # Setup test and train data.
    x_train, x_test, y_train, y_test = train_test_split(
        feature_vectors, labels, test_size=0.2, random_state=0, stratify=labels)

    # Regression part
    reg_accuracy, reg_cm, reg_coef = regression(
        x_train, x_test, y_train, y_test, nr_classes)
    print("Regression accuracy: " + str(reg_accuracy))

    if args.feature_importance:
        # If we want to plot the feature importance display it.
        plot_coef("Regression feature importance for author {author} "
                  "- {nr_classes} - class data".format(author=author.upper(),
                                                       nr_classes=nr_classes),
                  reg_coef, feature_names)

    # OVA part
    ova_accuracy, ova_cm, ova_coef_list = classify_ova(
        x_train, x_test, y_train, y_test)
    print("OVA accuracy: " + str(ova_accuracy))

    if args.feature_importance:
        # If we want to plot the feature importance display it.
        for i, ova_coef in enumerate(ova_coef_list):
            plot_coef("Ova feature importance for author {author} for class "
                      "{for_class} - {nr_classes} - class data"
                      .format(author=author.upper(),
                              for_class=str(i + 1),
                              nr_classes=nr_classes),
                      ova_coef, feature_names)

    # Write accuracies to file.
    with open(args.output, 'a') as f:
        f.write(f"{author},{nr_classes},reg,{reg_accuracy}\n")
        f.write(f"{author},{nr_classes},ova,{ova_accuracy}\n")


# Write header line to output file.
with open(args.output, 'w') as f:
    f.write('author,nr_classes,method,accuracy\n')

# For each author run both the 3 and 4 class case.
for author in data.subjective_sentences_files():
    print(f">>> Using dataset for author {author}")
    subjective_sentences_file = data.subjective_sentences_files()[author]
    three_class_labels_file = data.three_class_labels_files()[author]
    four_class_labels_file = data.four_class_labels_files()[author]

    with open(subjective_sentences_file) as f:
        subjective_sentences = f.readlines()

    config = configurations[args.configuration]
    __debug_log(">> Using config: {}".format(config))

    vectorizer = config['vectorizer']
    feature_vectors = vectorizer.fit_transform(subjective_sentences)
    feature_names = vectorizer.get_feature_names()

    print("# of features: " + str(len(feature_names)))

    print(">> With three class labels:")
    run(author, 3, feature_vectors, feature_names,
        __read_labels(three_class_labels_file))

    print(">> With four class labels:")
    run(author, 4, feature_vectors, feature_names,
        __read_labels(four_class_labels_file))
