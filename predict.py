import numpy as np

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, StratifiedKFold
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
from sklearn.svm import LinearSVR

from nltk.tokenize import word_tokenize
import argparse

from preprocessing import tokenize, remove_stopwords, lemmatize_words, Tokenizer
from configurations import configurations

subjective_sentences_files = {
    "dennis": "data/Dennis+Schwartz/subj.clean.Dennis+Schwartz",
    "james": "data/James+Berardinelli/subj.clean.James+Berardinelli",
    "scott": "data/Scott+Renshaw/subj.clean.Scott+Renshaw",
    "steve": "data/Steve+Rhodes/subj.clean.Steve+Rhodes",
}

three_class_labels_files = {
    "dennis": "data/Dennis+Schwartz/label.3class.clean.Dennis+Schwartz",
    "james": "data/James+Berardinelli/label.3class.clean.James+Berardinelli",
    "scott": "data/Scott+Renshaw/label.3class.clean.Scott+Renshaw",
    "steve": "data/Steve+Rhodes/label.3class.clean.Steve+Rhodes",
}

four_class_labels_files = {
    "dennis": "data/Dennis+Schwartz/label.4class.clean.Dennis+Schwartz",
    "james": "data/James+Berardinelli/label.4class.clean.James+Berardinelli",
    "scott": "data/Scott+Renshaw/label.4class.clean.Scott+Renshaw",
    "steve": "data/Steve+Rhodes/label.4class.clean.Steve+Rhodes",
}

parser = argparse.ArgumentParser()
parser.add_argument("--debug", action="store_true", help="Enable debug mode")
parser.add_argument("-c", "--configuration", choices=configurations, default='replicate-pang')
args = parser.parse_args()

print(args)

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
        svm_model = OneVsRestClassifier(SVC(kernel="linear", C=c)).fit(X_train, y_train)
    else:
        debug_log("> Running One-vs-All classifier with crossvalidation...")
        model_to_set = OneVsRestClassifier(SVC(kernel="linear"))
        params = {"estimator__C": np.logspace(-4, 2, 10)}

        svm_model = GridSearchCV(model_to_set, param_grid=params, n_jobs=-1)
        svm_model.fit(X_train, y_train)
        best_c = svm_model.best_estimator_.estimator.C
        debug_log("> Found best C value at " + str(best_c))

        svm_model = OneVsRestClassifier(SVC(kernel="linear", C=best_c)).fit(X_train, y_train)

    svm_predictions = svm_model.predict(X_test)

    accuracy = svm_model.score(X_test, y_test)

    cm = confusion_matrix(y_test, svm_predictions)
    debug_log("Confusion matrix:")
    debug_log(cm)

    return accuracy, cm


def regression(X_train, X_test, y_train, y_test, nr_classes, epsilon=-1, c=-1):
    if epsilon > -1 and c > -1:
        debug_log("> Running linear support vector regression without crossvalidation...")
        svm_model = LinearSVR(epsilon=epsilon, C=c).fit(X_train, y_train)
    else:
        debug_log("> Running linear support vector regression with crossvalidation...")
        params = [{"epsilon": np.logspace(-10, -1, 10)},
                  {"C": np.logspace(-4, 2, 10)}]
        svm_model = GridSearchCV(LinearSVR(), param_grid=params, n_jobs=-1)
        svm_model.fit(X_train, y_train)
        best_eps = svm_model.best_estimator_.epsilon
        best_c = svm_model.best_estimator_.C
        debug_log("> Found best epsilon value at " + str(best_eps))
        debug_log("> Found best C value at " + str(best_c))
        svm_model = LinearSVR(epsilon=best_eps, C=best_c)
        svm_model.fit(X_train, y_train)

    svm_predictions = svm_model.predict(X_test)
    rounded_predictions = np.round(svm_predictions)
    rounded_predictions = [nr_classes - 1 if x > nr_classes - 1 else x for x in rounded_predictions]
    accuracy = accuracy_score(rounded_predictions, y_test)

    cm = confusion_matrix(y_test, rounded_predictions)
    debug_log("Confusion matrix:")
    debug_log(cm)

    return accuracy, cm


def run(author, nr_classes, feature_vectors, labels):
    X_train, X_test, y_train, y_test = train_test_split(feature_vectors, labels, test_size=0.3,
                                                        random_state=0)
    reg_accuracy, reg_cm = regression(X_train, X_test, y_train, y_test, nr_classes)
    # for speed value for epsilon can be set to 0.00001
    print("Regression accuracy: " + str(reg_accuracy))
    ova_accuracy, ova_cm = classify_ova(X_train, X_test, y_train, y_test)
    # for speed value for C can be set to 0.005
    print("OVA accuracy: " + str(ova_accuracy))

    with open('results.csv', 'a') as f:
        f.write(f"{author},{nr_classes},reg,{reg_accuracy}\n")
        f.write(f"{author},{nr_classes},ova,{ova_accuracy}\n")

with open('results.csv', 'w') as f:
    f.write('author,nr_classes,method,accuracy\n')

for author_name in subjective_sentences_files:
    print(f">>> Using {author_name} dataset")
    subjective_sentences_file = subjective_sentences_files[author_name]
    three_class_labels_file = three_class_labels_files[author_name]
    four_class_labels_file = four_class_labels_files[author_name]

    subjective_sentences = []

    with open(subjective_sentences_file) as f:
        subjective_sentences = [line for line in f]

    config = configurations[args.configuration]
    debug_log(config)

    vectorizer = CountVectorizer(
        tokenizer=Tokenizer(config['tokenization_pipeline']).tokenize, 
        preprocessor=None, 
        binary=config['vectorizer'] == 'binary', 
        ngram_range=config['ngram_range'])
    feature_vectors = vectorizer.fit_transform(subjective_sentences)
    print("# of features: " + str(len(vectorizer.get_feature_names())))

    print(">> With three class labels:")
    run(author_name, 3, feature_vectors, read_labels(three_class_labels_file))

    print(">> With four class labels:")
    run(author_name, 4, feature_vectors, read_labels(four_class_labels_file))
