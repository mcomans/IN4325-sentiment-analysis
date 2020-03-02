from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

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
