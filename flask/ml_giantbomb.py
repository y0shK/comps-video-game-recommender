"""
Create a machine learning model using NLP that provides recommendations and clusters together suggestions
Methods: k-means clustering to put suggestions together in clusters
Evaluation: check to see if ground truth data and model output are in the same cluster.
    Are they within (<= N for some N s.t. 0 <= N <= 1) a certain threshold? If yes, 1, else 0
"""

"""
Proof of concept
"""

import requests_cache
import pdb
import json
import os
from dotenv import load_dotenv
import time
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

from tf_idf import get_fitted_train_matrix, get_unfitted_review_matrix
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import k_means

start_time = time.time()
load_dotenv()
session = requests_cache.CachedSession("giantbomb_cache")

GIANTBOMB_API_KEY = os.getenv('GIANTBOMB_API_KEY')
HEADERS = {'User-Agent': 'Video Game Recommender Comprehensive Project'}
QUERY = "cute artistic"

# https://www.giantbomb.com/api/documentation/#toc-0-17
game_url = "https://www.giantbomb.com/api/search/?api_key=" + GIANTBOMB_API_KEY + \
    "&format=json&query=" + QUERY + \
    "&resources=game" + \
    "&resource_type=game" 
    # resources = game details that should inform the results, while resource_type = game recommendation itself
game_resp = session.get(game_url, headers=HEADERS)
# pdb.set_trace()

game_json = json.loads(game_resp.text)
#game_results = game_json['results'][0]['name'] 

pdb.set_trace()

game_info = {}

for result in game_json['results']:
    print(result)
    print("-----")

    game_name = result['name']
    game_deck = result['deck']

    tokenized_name = word_tokenize(game_name)
    tokenized_deck = word_tokenize(game_deck)

    game_desc = result['description']
    game_screen_url = result['image']['screen_url']
    game_info[game_name] = {'name': tokenized_name, 'deck': tokenized_deck, 'desc': game_desc, 'screen_url': game_screen_url}

    # pdb.set_trace()



pdb.set_trace()

tf_idf_dict = {}

for k, v in game_info.items():
    tf_idf_dict[k] = {'name': v['name'], 'deck': v['deck']}

pdb.set_trace()

# TF-IDF name and deck,
# then use Euclidean distance as part of kmeans algorithm

# do k means test
#k = 3
#kmeans = k_means(n_clusters=k)
#y_pred = kmeans.fit_predict(get_fitted_train_matrix(''))