import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
from sklearn.svm import LinearSVR

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


def read_labels(labels_filepath):
    labels = []
    with open(labels_filepath) as f:
        for line in f:
            labels.append(int(line))

    return labels


def classify_ova(X_train, X_test, y_train, y_test):
    print("> Running One-vs-All classifier...")
    svm_model = OneVsRestClassifier(SVC(kernel="linear", C=1)).fit(X_train, y_train)

    svm_predictions = svm_model.predict(X_test)

    accuracy = svm_model.score(X_test, y_test)
    print(accuracy)

    cm = confusion_matrix(y_test, svm_predictions)
    print(cm)

    return accuracy


def regression(X_train, X_test, y_train, y_test):
    print("> Running linear support vector regression...")
    svm_model = LinearSVR(epsilon=0.1).fit(X_train, y_train)  # Not sure what epsilon should be

    accuracy = svm_model.score(X_test, y_test)
    print(accuracy)
    # Maybe we should create a confusion matrix as well for regression with rounded values
    return accuracy


def run(author, nr_classes, feature_vectors, labels):
    assert len(feature_vectors) == len(labels)

    X_train, X_test, y_train, y_test = train_test_split(feature_vectors, labels, test_size=0.5,
                                                        random_state=0)

    assert len(X_train) == len(y_train)
    assert len(X_test) == len(y_test)

    reg_accuracy = regression(X_train, X_test, y_train, y_test)
    ova_accuracy = classify_ova(X_train, X_test, y_train, y_test)

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
    word_to_index = {}

    with open(subjective_sentences_file) as f:
        for line in f:
            subjective_sentences.append(line)
            words = list(set(line.split(' ')))
            for word in words:
                word = word.strip()
                if word not in word_to_index:
                    word_to_index[word] = len(word_to_index)

    feature_vectors = []

    for sentence in subjective_sentences:
        feature_vector = np.zeros(len(word_to_index))
        words = list(set(sentence.split(' ')))
        for word in words:
            word = word.strip()
            feature_vector[word_to_index[word]] = 1

        feature_vectors.append(feature_vector)

    print(">> With three class labels")
    run(author_name, 3, feature_vectors, read_labels(three_class_labels_file))

    print(">> With four class labels")
    run(author_name, 4, feature_vectors, read_labels(four_class_labels_file))
