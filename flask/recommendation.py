"""
Supervised learning project to reverse engineer recommended games from GiantBomb API
"""
import requests_cache
import pdb
import json
import os
from dotenv import load_dotenv
import time
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import tensorflow as tf
import tensorflow_hub as hub

from get_api_info import get_giantbomb_game_info, get_gamespot_games, get_similar_games
from process_recs import process_text, check_for_valid_qualities, get_embedding_similarity
from calculate_metrics import calculate_confusion_matrix, calculate_average_pairs
import gensim.downloader

start_time = time.time()
load_dotenv()
session = requests_cache.CachedSession("recommendation_cache")

# http://www.gamespot.com/api/games/?api_key=<YOUR_API_KEY_HERE>
GAMESPOT_API_KEY = os.getenv('GAMESPOT_API_KEY')
HEADERS = {'User-Agent': 'Video Game Recommender Comprehensive Project'}
GIANTBOMB_API_KEY = os.getenv('GIANTBOMB_API_KEY')

# https://www.kaggle.com/datasets/dahlia25/metacritic-video-game-comments - use metacritic_game_info.csv to parse
csv_titles_df = pd.read_csv("flask/metacritic_game_info.csv")

"""
1. form a dataset by combining games from metacritic csv and GameSpot API
Get titles from each of the data sources, then get information from GiantBomb API call
"""
#csv_titles = list(set([i for i in csv_titles_df['Title']][0:15])) # 10 games
csv_titles = list(set([i for i in csv_titles_df['Title']][0:3]))
print(csv_titles[0])
pdb.set_trace()

# check for get_similar_games appropriate data structure
#csv_zero = get_similar_games(api_key=GIANTBOMB_API_KEY, query=csv_titles[0], headers=HEADERS, session=session)
#csv_zero = get_similar_games(api_key=GIANTBOMB_API_KEY, query="phoenix wright", headers=HEADERS, session=session)
#print("check if g dict shows up")
#pdb.set_trace()

dataset = {}
for title in csv_titles:
    query_dict = get_giantbomb_game_info(api_key=GIANTBOMB_API_KEY, query=title, headers=HEADERS,session=session)
    dataset = {**dataset, **query_dict}

pdb.set_trace()

#print(dataset)
print("investigate dataset")
#pdb.set_trace()

# get gamespot games
gamespot_games = get_gamespot_games(api_key=GAMESPOT_API_KEY, headers=HEADERS, game_count=2, session=session)
#print("look at gamespot games")
#pdb.set_trace()

dataset = {**dataset, **gamespot_games}

print("make sure all csv + gamespot games are recommended==0")
pdb.set_trace()

"""
2. Perform the recommendation step.
Use a pretrained word2vec model to generate recommendations based on the current loop iteration
Then compare the predictions to the actual values and calculate TP, FP, FN, TN
"""
# use a pretrained model
#model = gensim.downloader.load('glove-wiki-gigaword-50')

# set up tf universal encoder for cosine similarity comparisons on sentence embeddings
module_url = "https://tfhub.dev/google/universal-sentence-encoder/4"
model = hub.load(module_url)

print(type(model))
print("tf universal encoder set up!")

# for next improvement to algorithm, try:
# https://www.tensorflow.org/hub/tutorials/semantic_similarity_with_tf_hub_universal_encoder
# https://static.googleusercontent.com/media/research.google.com/en//pubs/archive/46808.pdf

# get thresholds
thresholds = list(np.linspace(0.1, 1, 100))
thresholds = [round(i, 2) for i in thresholds]

tokenizer = RegexpTokenizer(r'\w+')
stops = set(stopwords.words("english"))

# start algorithm
y_cos_sims = [] # becomes y_pred
y_true = []

model_sim_threshold = 0.75
game_recs = {}
total_pairs = []

# determine why 0s are occurring
cos_zero_dict = {'invalid': 0,
                 'model_sim': 0}

game_count = 1

for k1, v1 in dataset.items():
    similar_games_instance = get_similar_games(api_key=GIANTBOMB_API_KEY, query=k1, headers=HEADERS, session=session)
    all_recs = {**dataset, **similar_games_instance}
    #print("all_recs len: ", len(all_recs))

    print("dataset x all_recs")
    print(len(dataset), len(all_recs))
    print(len(dataset) * len(all_recs))

    # if any attribute (deck, description, genre, theme, franchise) is invalid, don't recommend
    vd1 = check_for_valid_qualities(v1['name'], v1['deck'], v1['description'],
        v1['genres'], v1['themes'], v1['franchises'])
    
    if vd1['deck'] and vd1['description']:
        v1_deck = process_text(v1['deck'])
        v1_desc = process_text(v1['description'])

    for k2, v2 in all_recs.items():
        model_sim = 0
        recs = {}

        # calculate y_true for metrics
        y_true.append(v2['recommended'])
        #print("len y_true", len(y_true))
        
        vd2 = check_for_valid_qualities(v2['name'], v2['deck'], v2['description'],
                                          v2['genres'], v2['themes'], v2['franchises'])
        
        valid = True in vd1.values() and True in vd2.values()

        if not valid:
            y_cos_sims.append(0)
            print("invalid rec")
            cos_zero_dict['invalid'] += 1
            #print("Len of y_cos_sims")
            #print(len(y_cos_sims))
            continue
        
        # start recommendation process
        if vd2['deck'] and vd2['description']:
            v2_deck = process_text(v2['deck'])
            v2_desc = process_text(v2['description'])
        
            model_sim = max(get_embedding_similarity(model, v1_deck, v2_deck), model_sim)
            if model_sim > model_sim_threshold:
                recs[k2] = model_sim

            model_sim = max(get_embedding_similarity(model, v1_desc, v2_desc), model_sim)
            if model_sim > model_sim_threshold:
                recs[k2] = model_sim

        for g in v2['genres']:
            if g in v1['genres']:
                model_sim = max(get_embedding_similarity(model, v1['genres'], v2['genres']), model_sim)
                recs[k2] = model_sim
        
        for g in v2['themes']:
            if g in v1['themes']:
                model_sim = max(get_embedding_similarity(model, v1['themes'], v2['themes']), model_sim)
                recs[k2] = model_sim
        
        for g in v2['franchises']:
            if g in v1['franchises']:
                model_sim = max(get_embedding_similarity(model, v1['franchises'], v2['franchises']), model_sim)
                recs[k2] = model_sim

        if model_sim > 0:
            y_cos_sims.append(model_sim)
        else:
            y_cos_sims.append(0)
            cos_zero_dict['model_sim'] += 1
        
    game_recs[k1] = recs
    
    print("game: ", k1)
    print("on " + str(game_count) + " of " + str(len(dataset)))
    game_count += 1
    total_pairs += [calculate_confusion_matrix(y_cos_sims, thresholds, y_true)]

print("after for loop")
pdb.set_trace()

print("recs")
print(recs)

#print("total_pairs")
#print(total_pairs)

avg_pairs = calculate_average_pairs(total_pairs)
avg_tvals = avg_pairs[0]
avg_fvals = avg_pairs[1]
threshold_vals = avg_pairs[2]

pdb.set_trace()

lin_x = np.linspace(0.0, 1.0, 11)
lin_y = np.linspace(0.0, 1.0, 11)
plt.plot(avg_fvals, avg_tvals)
plt.plot(lin_x, lin_y, label='linear')  # Plot some data on the (implicit) axes.
plt.xlabel("FPR")
plt.ylabel("TPR")
plt.title("ROC curve, averaged")
plt.show()

print("number of total cos sims entries:")
print(len(y_cos_sims))

plt.hist(y_cos_sims)
plt.show()

nonzero_cos_sims = [i for i in y_cos_sims if i > 0]

print("number of nonzero cos sims entries:")
print(len(nonzero_cos_sims))

print("max freq of nonzero cos sims")
# https://stackoverflow.com/questions/10797819/finding-the-mode-of-a-list
print(max(set(nonzero_cos_sims), key=nonzero_cos_sims.count))

plt.hist(nonzero_cos_sims)
plt.show()

plt.hist(nonzero_cos_sims, bins=100)
plt.show()

print("end time: ")
print(time.time() - start_time)

print("final pdb")
pdb.set_trace()