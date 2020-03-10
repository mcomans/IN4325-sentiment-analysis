data_path = 'data/'

authors = {
    'a': 'Steve+Rhodes',
    'b': 'Scott+Renshaw',
    'c': 'James+Berardinelli',
    'd': 'Dennis+Schwartz',
}


def generate_files_dict(label):
    """
    Generates the the file path for all authors with the given label prefix.
    :param label: The prefix for the particular file path to generate, e.g. subj.clean
    :return: File path, e.g. data/Dennis+Schwartz/subj.clean.Dennis+Schwartz
    """
    res = {}
    for a in authors:
        res[a] = f"{data_path}{authors[a]}/{label}.{authors[a]}"
    return res


def subjective_sentences_files(clean=True):
    """
    Gets file paths for all authors for the subjective sentences files.
    :param clean: Set to False if you need the original non-cleaned files.
    :return: Dictionary with author letter as key and file path as value.
    """
    if clean:
        return generate_files_dict('subj.clean')
    return generate_files_dict('subj')


def three_class_labels_files(clean=True):
    """
    Gets file paths for all authors for the three class labels files.
    :param clean: Set to False if you need the original non-cleaned files.
    :return: Dictionary with author letter as key and file path as value.
    """
    if clean:
        return generate_files_dict('label.3class.clean')
    return generate_files_dict('label.3class')


def four_class_labels_files(clean=True):
    """
    Gets file paths for all authors for the four class labels files.
    :param clean: Set to False if you need the original non-cleaned files.
    :return: Dictionary with author letter as key and file path as value.
    """
    if clean:
        return generate_files_dict('label.4class.clean')
    return generate_files_dict('label.4class')
