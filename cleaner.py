import re
import data

patterns = [
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


# Add the matched string to a list of removed matches for debugging. Returns empty since the match needs to be removed.
def save_and_remove(matchobj):
    removed_matches.append(matchobj[0])
    return ''


# Find duplicates in a list
def find_duplicates(lst, item):
    return [i for i, x in enumerate(lst) if x == item]


for author in data.subjective_sentences_files(False):
    sentences_file = data.subjective_sentences_files(False)[author]
    three_class_file = data.three_class_labels_files(False)[author]
    four_class_file = data.four_class_labels_files(False)[author]

    # Load files for cleaning. Class files are needed for removing duplicates.
    reviews = []
    cleaned_reviews = []
    with open(sentences_file) as f:
        for line in f:
            reviews.append(line)

    three_class_labels = []
    with open(three_class_file) as f:
        for line in f:
            three_class_labels.append(line)

    four_class_labels = []
    with open(four_class_file) as f:
        for line in f:
            four_class_labels.append(line)

    # Remove duplicates because yes, there are duplicates
    duplicates = dict((x, find_duplicates(reviews, x)) for x in set(reviews) if reviews.count(x) > 1)
    duplicate_idxs = [loc[0] for loc in duplicates.values()]
    for idx in sorted(duplicate_idxs, reverse=True):
        del reviews[idx]
        del three_class_labels[idx]
        del four_class_labels[idx]

    # Match sentences with regex patterns above to remove explicit ratings etc.
    for r in reviews:
        cleaned = r
        for p in patterns:
            cleaned = re.sub(p, save_and_remove, cleaned)
        cleaned_reviews.append(cleaned)

    # Write cleaned data to files
    with open(data.subjective_sentences_files()[author], 'w') as f:
        for line in cleaned_reviews:
            f.write(line)

    with open(data.three_class_labels_files()[author], 'w') as f:
        for line in three_class_labels:
            f.write(line)

    with open(data.four_class_labels_files()[author], 'w') as f:
        for line in four_class_labels:
            f.write(line)

if DEBUG:
    for m in removed_matches:
        print(f"{m}\n")
