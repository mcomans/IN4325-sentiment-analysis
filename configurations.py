from preprocessing import tokenize, remove_stopwords, lemmatize_words

configurations = {
    'replicate-pang': {
        'ngram_range': [1,1],
        'tokenization_pipeline': [tokenize],
        'vectorizer': 'binary'
    }
}