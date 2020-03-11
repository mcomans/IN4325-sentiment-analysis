# This file describes the process of data cleaning using a set of regexps to
# clear out indicators of a rating that could potentially affect the ML models.

import re
import data

PATTERNS = [
    # Dennis Schwartz
    r' ?rating : \* .*',

    # Steve Rhodes
    r' ?\* \* \* \* = .*',
    r' ?(1\/2 )?((\* )+(1\/2 )?(stars)?|of a star)',
    r' ?1\/2 (stars?|of a star)+',
    r' ?(mild |one |two |three )?thumbs (way )?(up|down|sideways)',
]
DEBUG = False
removed_matches = []


def __save_and_remove(mach_obj):
    """
    Add the matched string to a list of removed matches for debugging.
    Returns empty since the match needs to be removed.
    """
    removed_matches.append(mach_obj[0])
    return ''


def __find_duplicates(lst, item):
    """Find duplicates of a given item in a given list excluding the
    original."""
    return [i for i, x in enumerate(lst) if x == item][1:]


def __read_data_file(file):
    """
    Returns the data entries in a given file.
    The entries are assumed to stored as be one per line.
    """
    with open(file) as f:
        return f.readlines()


def __write_data_file(file, data):
    """
    Writes a set of data to the specified file.
    """
    with open(file, 'w') as f:
        f.writelines(data)


# The data is cleaned on a per author basis.
# Once loaded, duplicates are removed and matches with the regexps defined in
# patterns are removed from each review.
for author in data.subjective_sentences_files(clean=False):
    sentences_file = data.subjective_sentences_files(clean=False)[author]
    three_class_file = data.three_class_labels_files(clean=False)[author]
    four_class_file = data.four_class_labels_files(clean=False)[author]

    # Load files for cleaning. Class files are needed for removing duplicates.
    reviews = __read_data_file(sentences_file)
    three_class_labels = __read_data_file(three_class_file)
    four_class_labels = __read_data_file(four_class_file)

    # Remove duplicates because yes, there are duplicates
    duplicates = []
    for x in set(reviews):
        if reviews.count(x) > 1:
            duplicates.extend(__find_duplicates(reviews, x))
    for idx in sorted(duplicates, reverse=True):
        del reviews[idx]
        del three_class_labels[idx]
        del four_class_labels[idx]

    cleaned_reviews = []
    # Match sentences with regex patterns above to remove explicit ratings etc.
    for rev in reviews:
        cleaned_rev = rev
        for p in PATTERNS:
            cleaned_rev = re.sub(p, __save_and_remove, cleaned_rev)
        cleaned_reviews.append(cleaned_rev)

    # Write cleaned data to files
    __write_data_file(data.subjective_sentences_files(clean=True)[author],
                      cleaned_reviews)
    __write_data_file(data.three_class_labels_files(clean=True)[author],
                      three_class_labels)
    __write_data_file(data.four_class_labels_files(clean=True)[author],
                      four_class_labels)


# If DEBUG is enabled we will print out all the matches that were removed.
if DEBUG:
    for m in removed_matches:
        print(f"{m}\n")
