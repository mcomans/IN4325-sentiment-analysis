import numpy as np

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
from sklearn.svm import LinearSVR

import argparse

from configurations import configurations
import data

parser = argparse.ArgumentParser()
parser.add_argument("--debug", action="store_true", help="Enable debug mode")
parser.add_argument("-c", "--configuration", choices=configurations,
                    default='replicate-pang', help="Configuration preset")
parser.add_argument("-o", "--output", help="Output filename",
                    default="results.csv")
args = parser.parse_args()


def debug_log(text):
    if (args.debug):
        print(text)


def read_labels(labels_filepath):
    labels = []
    with open(labels_filepath) as f:
        for line in f:
            labels.append(int(line))

    return labels


def classify_ova(X_train, X_test, y_train, y_test, c=-1):
    if c > - 1:
        debug_log("> Running One-vs-All classifier without crossvalidation...")
        svm_model = OneVsRestClassifier(
            SVC(kernel="linear", C=c)).fit(X_train, y_train)
    else:
        debug_log("> Running One-vs-All classifier with crossvalidation...")
        model_to_set = OneVsRestClassifier(SVC(kernel="linear"))
        params = {"estimator__C": np.logspace(-4, 2, 6)exi}

        svm_model = GridSearchCV(model_to_set, param_grid=params, n_jobs=-1, cv=10)
        svm_model.fit(X_train, y_train)
        best_c = svm_model.best_estimator_.estimator.C
        debug_log("> Found best C value at " + str(best_c))

        svm_model = OneVsRestClassifier(
            SVC(kernel="linear", C=best_c)).fit(X_train, y_train)

    svm_predictions = svm_model.predict(X_test)

    accuracy = svm_model.score(X_test, y_test)

    cm = confusion_matrix(y_test, svm_predictions)
    debug_log("Confusion matrix:")
    debug_log(cm)

    return accuracy, cm


def regression(X_train, X_test, y_train, y_test, nr_classes, epsilon=-1, c=-1):
    if epsilon > -1 and c > -1:
        debug_log(
            "> Running linear support vector regression without crossvalidation...")
        svm_model = LinearSVR(epsilon=epsilon, C=c).fit(X_train, y_train)
    else:
        debug_log(
            "> Running linear support vector regression with crossvalidation...")
        params = [{"epsilon": np.logspace(-10, -1, 5)},
                  {"C": np.logspace(-4, 2, 6)}]
        svm_model = GridSearchCV(LinearSVR(), param_grid=params, n_jobs=-1, cv=8)
        svm_model.fit(X_train, y_train)
        best_eps = svm_model.best_estimator_.epsilon
        best_c = svm_model.best_estimator_.C
        debug_log("> Found best epsilon value at " + str(best_eps))
        debug_log("> Found best C value at " + str(best_c))
        svm_model = LinearSVR(epsilon=best_eps, C=best_c)
        svm_model.fit(X_train, y_train)

    svm_predictions = svm_model.predict(X_test)
    rounded_predictions = np.round(svm_predictions)
    rounded_predictions = [nr_classes - 1 if x >
                           nr_classes - 1 else x for x in rounded_predictions]
    accuracy = accuracy_score(rounded_predictions, y_test)

    cm = confusion_matrix(y_test, rounded_predictions)
    debug_log("Confusion matrix:")
    debug_log(cm)

    return accuracy, cm


def run(author, nr_classes, feature_vectors, labels):
    X_train, X_test, y_train, y_test = train_test_split(
        feature_vectors, labels, test_size=0.2, random_state=0, stratify=labels)
    reg_accuracy, reg_cm = regression(
        X_train, X_test, y_train, y_test, nr_classes)
    # for speed value for epsilon can be set to 0.00001
    print("Regression accuracy: " + str(reg_accuracy))
    ova_accuracy, ova_cm = classify_ova(X_train, X_test, y_train, y_test)
    # for speed value for C can be set to 0.005
    print("OVA accuracy: " + str(ova_accuracy))

    with open(args.output, 'a') as f:
        f.write(f"{author},{nr_classes},reg,{reg_accuracy}\n")
        f.write(f"{author},{nr_classes},ova,{ova_accuracy}\n")


with open(args.output, 'w') as f:
    f.write('author,nr_classes,method,accuracy\n')

for author in data.subjective_sentences_files():
    print(f">>> Using dataset for author {author}")
    subjective_sentences_file = data.subjective_sentences_files()[author]
    three_class_labels_file = data.three_class_labels_files()[author]
    four_class_labels_file = data.four_class_labels_files()[author]

    subjective_sentences = []

    with open(subjective_sentences_file) as f:
        subjective_sentences = [line for line in f]

    config = configurations[args.configuration]
    debug_log(config)

    vectorizer = config['vectorizer']
    feature_vectors = vectorizer.fit_transform(subjective_sentences)

    print("# of features: " + str(len(vectorizer.get_feature_names())))

    print(">> With three class labels:")
    run(author, 3, feature_vectors, read_labels(three_class_labels_file))

    print(">> With four class labels:")
    run(author, 4, feature_vectors, read_labels(four_class_labels_file))
