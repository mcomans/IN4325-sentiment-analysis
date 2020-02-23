import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
from sklearn.svm import LinearSVR

subjective_sentences_file = "data/Steve+Rhodes/subj.Steve+Rhodes"
four_class_labels_file = "data/Steve+Rhodes/label.4class.Steve+Rhodes"

subjective_sentences = []
word_to_index = {}
labels = []

with open(subjective_sentences_file) as f:
  for line in f:
    subjective_sentences.append(line)
    words = list(set(line.split(' ')))
    for word in words:
      word = word.strip()
      if word not in word_to_index:
        word_to_index[word] = len(word_to_index)

with open(four_class_labels_file) as f:
  for line in f:
    labels.append(int(line))

feature_vectors = []

for sentence in subjective_sentences:
  feature_vector = np.zeros(len(word_to_index))
  words = list(set(sentence.split(' ')))
  for word in words:
    word = word.strip()
    feature_vector[word_to_index[word]] = 1

  feature_vectors.append(feature_vector)

assert len(feature_vectors) == len(labels)

X_train, X_test, y_train, y_test = train_test_split(feature_vectors, labels, test_size=0.5, random_state = 0)

assert len(X_train) == len(y_train)
assert len(X_test) == len(y_test)

# One-vs-All classifier
# svm_model = OneVsRestClassifier(SVC(kernel="linear", C=1)).fit(X_train, y_train)

# Linear, epsilon-intensive support vector regression
svm_model = LinearSVR(epsilon=0.1).fit(X_train, y_train) # Not sure what epsilon should be

svm_predictions = svm_model.predict(X_test)

accuracy = svm_model.score(X_test, y_test)
print(accuracy)

# cm = confusion_matrix(y_test, svm_predictions)
# print(cm)
