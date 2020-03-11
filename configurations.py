from preprocessing import remove_stopwords, lemmatize_words, Tokenizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

configurations = {
    'unigrams': {
        'vectorizer': CountVectorizer(
            binary=True,
            tokenizer=Tokenizer().tokenize,
            ngram_range=[1, 1])
    },
    'unigrams-tfidf': {
        'vectorizer': TfidfVectorizer(
            tokenizer=Tokenizer().tokenize,
            ngram_range=[1, 1])
    },
    'unigrams-lemmatization': {
        'vectorizer': CountVectorizer(
            binary=True,
            tokenizer=Tokenizer([lemmatize_words]).tokenize,
            ngram_range=[1, 1])
    },
    'unigrams-lemmatization-tfidf': {
        'vectorizer': TfidfVectorizer(
            tokenizer=Tokenizer([lemmatize_words]).tokenize,
            ngram_range=[1, 1])
    },
    'unigrams-lemmatization-stopwords': {
        'vectorizer': CountVectorizer(
            binary=True,
            tokenizer=Tokenizer([lemmatize_words, remove_stopwords]).tokenize,
            ngram_range=[1, 1])
    },
    'unigrams-bigrams': {
        'vectorizer': CountVectorizer(
            binary=True,
            tokenizer=Tokenizer().tokenize,
            ngram_range=[1, 2])
    },
    'unigrams-bigrams-tfidf': {
        'vectorizer': TfidfVectorizer(
            tokenizer=Tokenizer().tokenize,
            ngram_range=[1, 2])
    },
    'unigrams-bigrams-lemmatization': {
        'vectorizer': CountVectorizer(
            binary=True,
            tokenizer=Tokenizer([lemmatize_words]).tokenize,
            ngram_range=[1, 2])
    },
    'unigrams-bigrams-lemmatization-tfidf': {
        'vectorizer': TfidfVectorizer(
            tokenizer=Tokenizer([lemmatize_words]).tokenize,
            ngram_range=[1, 2])
    },
    'unigrams-bigrams-lemmatization-stopwords': {
        'vectorizer': CountVectorizer(
            binary=True,
            tokenizer=Tokenizer([lemmatize_words, remove_stopwords]).tokenize,
            ngram_range=[1, 2])
    },
}
