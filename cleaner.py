import re

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

cleaned_sentences_files = {
    "dennis": "data/Dennis+Schwartz/subj.clean.Dennis+Schwartz",
    "james": "data/James+Berardinelli/subj.clean.James+Berardinelli",
    "scott": "data/Scott+Renshaw/subj.clean.Scott+Renshaw",
    "steve": "data/Steve+Rhodes/subj.clean.Steve+Rhodes",
}

cleaned_three_class_files = {
    "dennis": "data/Dennis+Schwartz/label.3class.clean.Dennis+Schwartz",
    "james": "data/James+Berardinelli/label.3class.clean.James+Berardinelli",
    "scott": "data/Scott+Renshaw/label.3class.clean.Scott+Renshaw",
    "steve": "data/Steve+Rhodes/label.3class.clean.Steve+Rhodes",
}

cleaned_four_class_files = {
    "dennis": "data/Dennis+Schwartz/label.4class.clean.Dennis+Schwartz",
    "james": "data/James+Berardinelli/label.4class.clean.James+Berardinelli",
    "scott": "data/Scott+Renshaw/label.4class.clean.Scott+Renshaw",
    "steve": "data/Steve+Rhodes/label.4class.clean.Steve+Rhodes",
}

patterns = [
    # Dennis Schwartz
    r' ?rating : \* .*',

    # Steve Rhodes
    r' ?\* \* \* \* = .*',
    r' ?(1\/2)?((\* ?)+|of a star|1\/2) ?',
]

DEBUG = True

removed_matches = []


# Add the matched string to a list of removed matches for debugging. Returns empty since the match needs to be removed.
def save_and_remove(matchobj):
    removed_matches.append(matchobj[0])
    return ''


# Find duplicates in a list
def find_duplicates(lst, item):
    return [i for i, x in enumerate(lst) if x == item]


for author_name in subjective_sentences_files:
    sentences_file = subjective_sentences_files[author_name]
    three_class_file = three_class_labels_files[author_name]
    four_class_file = four_class_labels_files[author_name]

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
    with open(cleaned_sentences_files[author_name], 'w') as f:
        for line in cleaned_reviews:
            f.write(line)

    with open(cleaned_three_class_files[author_name], 'w') as f:
        for line in three_class_labels:
            f.write(line)

    with open(cleaned_four_class_files[author_name], 'w') as f:
        for line in four_class_labels:
            f.write(line)

if DEBUG:
    for m in removed_matches:
        print(f"{m}\n")
