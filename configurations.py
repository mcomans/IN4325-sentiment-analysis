from preprocessing import tokenize, remove_stopwords, lemmatize_words, Tokenizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

configurations = {
    'replicate-pang': {
        'vectorizer': CountVectorizer(
            binary=True, tokenizer=Tokenizer([tokenize]).tokenize,
            ngram_range=[1, 1])
    },
    'unigrams-bigrams': {
        'vectorizer': CountVectorizer(
            binary=True, tokenizer=Tokenizer([tokenize]).tokenize,
            ngram_range=[1, 2])
    },
    'unigrams-bigrams-freq': {
        'vectorizer': CountVectorizer(
            tokenizer=Tokenizer([tokenize]).tokenize,
            ngram_range=[1, 2])
    },
    'unigrams-bigrams-tfidf': {
        'vectorizer': TfidfVectorizer(
            tokenizer=Tokenizer([tokenize]).tokenize,
            ngram_range=[1, 2])
    },
    'unigrams-lemmatization-stopwords': {
        'vectorizer': CountVectorizer(
            tokenizer=Tokenizer([tokenize, lemmatize_words, remove_stopwords]).tokenize, ngram_range=[1, 1], binary=True)
    },
    'unigrams-bigrams-lemmatization-stopwords': {
        'vectorizer': CountVectorizer(tokenizer=Tokenizer([tokenize, lemmatize_words, remove_stopwords]).tokenize, ngram_range=[1,2], binary=True)
    },
    'unigrams-tfidf': {
        'vectorizer': TfidfVectorizer(tokenizer=Tokenizer([tokenize]).tokenize, ngram_range=[1, 1])
    },
    'unigrams-lemmatization': {
        'vectorizer': CountVectorizer(
            tokenizer=Tokenizer([tokenize, lemmatize_words]).tokenize,
            ngram_range=[1, 1], binary=True)
    }
}
