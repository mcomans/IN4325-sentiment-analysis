import numpy as np

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
from sklearn.svm import LinearSVR

from nltk.tokenize import word_tokenize

from preprocessing import tokenize, remove_stopwords, lemmatize_words

subjective_sentences_files = {
    "dennis": "data/Dennis+Schwartz/subj.Dennis+Schwartz",
    "james": "data/James+Berardinelli/subj.James+Berardinelli",
    "scott": "data/Scott+Renshaw/subj.Scott+Renshaw",
    "steve": "data/Steve+Rhodes/subj.Steve+Rhodes",
}

three_class_labels_files = {
    "dennis": "data/Dennis+Schwartz/label.3class.Dennis+Schwartz",
    "james": "data/James+Berardinelli/label.3class.James+Berardinelli",
    "scott": "data/Scott+Renshaw/label.3class.Scott+Renshaw",
    "steve": "data/Steve+Rhodes/label.3class.Steve+Rhodes",
}

four_class_labels_files = {
    "dennis": "data/Dennis+Schwartz/label.4class.Dennis+Schwartz",
    "james": "data/James+Berardinelli/label.4class.James+Berardinelli",
    "scott": "data/Scott+Renshaw/label.4class.Scott+Renshaw",
    "steve": "data/Steve+Rhodes/label.4class.Steve+Rhodes",
}

parameter_configs = {
    "unigrams": { 'ngram_range': [1,1] },
    "bigrams": { 'ngram_range': [2,2]},
    "bigrams and unigrams": {'ngram_range': [1,2]}
}

def read_labels(labels_filepath):
    labels = []
    with open(labels_filepath) as f:
        for line in f:
            labels.append(int(line))

    return labels


def classify_ova(X_train, X_test, y_train, y_test, c):
    if c > - 1:
        print("> Running One-vs-All classifier without crossvalidation...")
        svm_model = OneVsRestClassifier(SVC(kernel="linear", C=c)).fit(X_train, y_train)
    else:
        print("> Running One-vs-All classifier with crossvalidation...")
        params = [{ "kernel": ["linear"], "C": np.logspace(-6, 1, 25)}]
        svm_model = GridSearchCV(SVC(), param_grid=params, n_jobs=-1)
        svm_model.fit(X_train, y_train)        
        print("> Found best C value at " + str(svm_model.best_estimator_.C))

    svm_predictions = svm_model.predict(X_test)

    accuracy = svm_model.score(X_test, y_test)
    print(accuracy)

    cm = confusion_matrix(y_test, svm_predictions)
    print(cm)

    return accuracy


def regression(X_train, X_test, y_train, y_test, epsilon):
    if epsilon > - 1:
        print("> Running linear support vector regression without crossvalidation...")
        svm_model = LinearSVR(epsilon=epsilon).fit(X_train, y_train)  # Not sure what epsilon should be
    else:
        print("> Running linear support vector regression with crossvalidation...")
        params = [{"epsilon": np.logspace(-6, 1, 25)}]
        svm_model = GridSearchCV(LinearSVR(), param_grid=params, n_jobs=-1)
        svm_model.fit(X_train, y_train)
        print("> Found best C value at " + str(svm_model.best_estimator_.C))

    svm_predictions = svm_model.predict(X_test)
    rounded_predictions = np.round(svm_predictions)
    accuracy = accuracy_score(rounded_predictions, y_test)
    print(accuracy)

    cm = confusion_matrix(y_test, rounded_predictions)
    print(cm)

    return accuracy

def run(author, nr_classes, feature_vectors, labels):
    # assert len(feature_vectors) == len(labels)

    X_train, X_test, y_train, y_test = train_test_split(feature_vectors, labels, test_size=0.5,
                                                        random_state=0)

    # assert len(X_train) == len(y_train)
    # assert len(X_test) == len(y_test)

    reg_accuracy = regression(X_train, X_test, y_train, y_test, -1)
    ova_accuracy = classify_ova(X_train, X_test, y_train, y_test, -1)

    with open('results.csv', 'a') as f:
        f.write(f"{author},{nr_classes},reg,{reg_accuracy}\n")
        f.write(f"{author},{nr_classes},ova,{ova_accuracy}\n")

def preprocessor(text):
    return text

def tokenizer(text):
    tokenized_words = tokenize(text)
    filtered_words = remove_stopwords(tokenized_words)
    lemmatized_words = lemmatize_words(filtered_words)
    return lemmatized_words

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

    vectorizer = CountVectorizer(tokenizer=tokenizer, preprocessor=None, binary=True, ngram_range=[1,2])
    feature_vectors = vectorizer.fit_transform(subjective_sentences)

    print(">> With three class labels")
    run(author_name, 3, feature_vectors, read_labels(three_class_labels_file))

    print(">> With four class labels")
    run(author_name, 4, feature_vectors, read_labels(four_class_labels_file))
