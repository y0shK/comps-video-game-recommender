"""
tf_idf.py

Create custom methods to implement the tf-idf algorithm using sklearn's TfidfVectorizer.

get_fitted_train_matrix(["query plaintext"])
- takes the user query plaintext string in list format 
- returns a tf-idf matrix (fitted and transformed to the query)

get_unfitted_review_matrix(["query plaintext"], ["review plaintext"])
- takes the user query plaintext string and review plaintext string in list format
- returns a tf-idf matrix (fitted to user query and transformed to review)
"""

from sklearn.feature_extraction.text import TfidfVectorizer

# user query matrix
"""
get_fitted_train_matrix
Arguments: list of strings to train tf-idf vectorizer
Returns: list containing vectorizer and matrix (fitted and transformed)
"""
def get_fitted_train_matrix(corpus):

    # transform the query into a tf-idf vector
    v = TfidfVectorizer(ngram_range=(1,2), stop_words='english')
    tfidf_matrix = v.fit_transform(corpus) # fit_transform is fitting and transforming on corpus
    return [v, tfidf_matrix]

# review matrix per game
"""
get_unfitted_train_matrix
Arguments: list of strings to train tf-idf vectorizer
Returns: list containing vectorizer and matrix (only transformed)
"""
def get_unfitted_review_matrix(fit_corpus, transform_corpus):
    
    # transform the review into a tf-idf vector
    v = TfidfVectorizer(ngram_range=(1,2), stop_words='english')
    v.fit(fit_corpus) # fit on original corpus (user query)
    matrix = v.transform(transform_corpus) # transform on new corpus (review)
    return [v, matrix]