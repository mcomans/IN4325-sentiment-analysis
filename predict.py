from sklearn.model_selection import train_test_split

import argparse
import logging

from configurations import configurations
from models.Regression import regression
from models.OneVsAll import classify_ova
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


if args.debug:
    logging.getLogger().setLevel(level=10)
    logging.debug("DEBUG logging enabled")
else:
    logging.getLogger().setLevel(level=30)


def __read_labels(labels_filepath):
    """Read labels from file as integers."""
    with open(labels_filepath) as f:
        return [int(line) for line in f.readlines()]


def __run(author, nr_classes, feature_vectors, feature_names, labels):
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
    logging.debug(">> Using config: {}".format(config))

    vectorizer = config['vectorizer']
    feature_vectors = vectorizer.fit_transform(subjective_sentences)
    feature_names = vectorizer.get_feature_names()

    print("# of features: " + str(len(feature_names)))

    print(">> With three class labels:")
    __run(author, 3, feature_vectors, feature_names,
          __read_labels(three_class_labels_file))

    print(">> With four class labels:")
    __run(author, 4, feature_vectors, feature_names,
          __read_labels(four_class_labels_file))
