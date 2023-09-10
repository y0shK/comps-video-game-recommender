"""
game_review_clusters.py
Get game review clusters for unsupervised learning k-means clustering approach
No train/test like classification problem required. Instead, get the clusters, validate with k-means metrics
Then use review clusters to figure out how games should be clustered, which in turn is provided to the user
"""

# following https://scikit-learn.org/stable/auto_examples/text/plot_document_clustering.html
# currently working on proof of concept for review clustering

"""
1. Imports and setup to get reviews from API
"""
import requests_cache
import pdb
import json
import os
from dotenv import load_dotenv
import time
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import numpy as np

start_time = time.time()
load_dotenv()
session = requests_cache.CachedSession("review_clusters_cache")

# http://www.gamespot.com/api/games/?api_key=<YOUR_API_KEY_HERE>
GAMESPOT_API_KEY = os.getenv('GAMESPOT_API_KEY')
HEADERS = {'User-Agent': 'Video Game Recommender Comprehensive Project'}

from backend_gamespot import get_review_info, get_reviews

reviews = get_reviews(api_key=GAMESPOT_API_KEY, headers=HEADERS, review_count=20) 

review_list = []
for k, v in reviews.items():
    review_list.append(v['body'])

# pdb.set_trace()

"""
2. Vectorize dataset of reviews
"""

from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(
    max_df=0.5,
    min_df=5,
    stop_words="english",
)
t0 = time.time()
X_tfidf = vectorizer.fit_transform(review_list)

# pdb.set_trace()

print(f"vectorization done in {time.time() - t0:.3f} s")
print(f"n_samples: {X_tfidf.shape[0]}, n_features: {X_tfidf.shape[1]}")

print(f"{X_tfidf.nnz / np.prod(X_tfidf.shape):.3f}")

"""
3. Start kmeans clustering
"""

from sklearn.cluster import KMeans

# TODO use elbow method to find best k
k = 5 # arbitrary

for seed in range(5):
    kmeans = KMeans(
        n_clusters=k,
        max_iter=100,
        n_init=1,
        random_state=seed,
    ).fit(X_tfidf)
    cluster_ids, cluster_sizes = np.unique(kmeans.labels_, return_counts=True)
    print(f"Number of elements assigned to each cluster: {cluster_sizes}")
print()
#print(
 #   "True number of documents in each category according to the class labels: "
  #  f"{category_sizes}"
#)

kmeans = KMeans(
    n_clusters=k,
    max_iter=100,
    n_init=5,
)

# TODO get labels so that metrics can be calculated
# TODO evaluation

kmeans.fit(X_tfidf)
y_pred = kmeans.fit_predict(X_tfidf)
clusters = kmeans.cluster_centers_

# pdb.set_trace()

"""
4. perform LSA on TF-IDF vector to reduce dimensionality
"""

from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer

lsa = make_pipeline(TruncatedSVD(n_components=100), Normalizer(copy=False))
t0 = time.time()
X_lsa = lsa.fit_transform(X_tfidf)
explained_variance = lsa[0].explained_variance_ratio_.sum()

print(f"LSA done in {time.time() - t0:.3f} s")
print(f"Explained variance of the SVD step: {explained_variance * 100:.1f}%")

# pdb.set_trace()

"""
5. Play around with cluster results
"""

# inverse_transform does not provide the right matrix dimensions; change to transform to transpose the transpose
original_space_centroids = lsa[0].transform(kmeans.cluster_centers_)
order_centroids = original_space_centroids.argsort()[:, ::1]
terms = vectorizer.get_feature_names_out()

pdb.set_trace()

for i in range(k):
    print(f"Cluster {i}: ", end="")
    for ind in order_centroids[i, :100]:
        #if terms[ind].isalpha():

        """
        In proof of concept, add an offset so that the string numbers '2' don't show up - words show instead
        TODO Later, do proper preprocessing so that only (relatively informatic) words show up
        """
        print(f"{terms[ind + 2000]} ", end="") # add some offset so that the words show up
        # print(terms[ind])
    print('')

pdb.set_trace()