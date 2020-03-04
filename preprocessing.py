from nltk.corpus import stopwords, wordnet
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.tag import pos_tag

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

def to_wordnet(tag):
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

def lemmatize_words(tokens):
    """Uses WordNet lemmatizer to lemmatize list of tokens"""
    tagged_tokens = [(x[0], to_wordnet(x[1])) for x in pos_tag(tokens)]
    # Use wordnet lemmatizer
    wnl = WordNetLemmatizer()
    # Return lemmatized tokens
    lemmatized_tokens = [wnl.lemmatize(x[0], x[1]) for x in tagged_tokens]
    return lemmatized_tokens
