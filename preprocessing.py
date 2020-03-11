# This file contains functions to preprocess the data like the use of a
# tokenizer

from nltk.corpus import stopwords, wordnet
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.tag import pos_tag


class Tokenizer:
    """Tokenizer class that is able to handle a set of tokanisation steps."""

    def __init__(self, steps=None):
        """
        Initialise the tokenizer with a set of steps to apply when given a piece
        of text.

        Each step should take as input a list of words and return the
        processed output as a list of words.
        """
        if steps is None:
            steps = []
        self.steps = steps

    def tokenize(self, text):
        """First tokenize the text and then apply the provided steps for this
        Tokenizer. """
        # Use NLTK word tokenizer as base.
        result = word_tokenize(text)
        for step in self.steps[:]:
            result = step(result)
        return result


def __to_wordnet(tag):
    """Given a tag convert it to the respective wordnet tag."""
    if tag.startswith('J'):
        return wordnet.ADJ
    elif tag.startswith('V'):
        return wordnet.VERB
    elif tag.startswith('N'):
        return wordnet.NOUN
    elif tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN


# Below are two optional steps that could be given to the Tokenizer.
def remove_stopwords(words):
    """Removes stopwords from list of words."""
    # Use english stopwords corpus
    stopwords_en = set(stopwords.words('english'))
    # Filter and return non-stopwords
    return [word for word in words if word not in stopwords_en]


def lemmatize_words(tokens):
    """Uses WordNet lemmatizer to lemmatize list of tokens."""
    tagged_tokens = [(x[0], __to_wordnet(x[1])) for x in pos_tag(tokens)]
    # Use wordnet lemmatizer
    wnl = WordNetLemmatizer()
    # Return lemmatized tokens
    lemmatized_tokens = [wnl.lemmatize(x[0], x[1]) for x in tagged_tokens]
    return lemmatized_tokens
