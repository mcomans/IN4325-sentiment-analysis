from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

class Tokenizer:
    def __init__(self, steps):
        self.steps = steps

    def tokenize(self, text):
        if (len(self.steps) < 1):
            return text
        steps_cp = self.steps[:]
        result = steps_cp.pop(0)(text)
        for step in steps_cp:
            result = step(result)
        return result

def tokenize(text):
    """Tokenizes text to words"""
    # Use NLTK word tokanizer
    return word_tokenize(text)

def remove_stopwords(words):
    """Removes stopwords from list of words"""
    # Use english stopwords corpus
    stopwords_en = set(stopwords.words('english'))
    # Filter and return non-stopwords
    return [word for word in words if word not in stopwords_en]

def lemmatize_words(words):
    """Uses WordNet lemmatizer to lemmatize list of words"""
    # Use wordnet lemmatizer
    wnl = WordNetLemmatizer()
    # Return lemmatized words
    return [wnl.lemmatize(word) for word in words]
