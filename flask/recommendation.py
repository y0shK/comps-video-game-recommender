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

from get_api_info import get_giantbomb_game_info, get_gamespot_games
from process_recs import process_text, check_for_valid_qualities
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
csv_titles = list(set([i for i in csv_titles_df['Title']][0:3])) # 3 games
print(csv_titles[0])
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
gamespot_games = get_gamespot_games(api_key=GAMESPOT_API_KEY, headers=HEADERS, game_count=1)
#print("look at gamespot games")
#pdb.set_trace()

dataset = {**dataset, **gamespot_games}
pdb.set_trace()

"""
2. Perform the recommendation step.
Use a pretrained word2vec model to generate recommendations based on the current loop iteration
Then compare the predictions to the actual values and calculate TP, FP, FN, TN
"""
# use a pretrained model
model = gensim.downloader.load('glove-wiki-gigaword-50')

# get thresholds
thresholds = list(np.linspace(0.1, 1, 100))
thresholds = [round(i, 2) for i in thresholds]

tokenizer = RegexpTokenizer(r'\w+')
stops = set(stopwords.words("english"))

# start algorithm
y_cos_sims = []
model_sim_threshold = 0.9
recs = {}
total_pairs = []
y_true = []
for k1, v1 in dataset.items():

    # calculate actual values
    y_true = [v1['recommended'] for k1, v1 in dataset.items()]
    print(len(y_true))

    for k2, v2 in dataset.items():
        model_sim = 0

        # if any attribute (deck, description, genre, theme, franchise) is invalid, don't recommend
        valid1 = check_for_valid_qualities(v1['name'], v1['deck'], v1['description'],
                                          v1['genres'], v1['themes'], v1['franchises'])
        
        valid2 = check_for_valid_qualities(v2['name'], v2['deck'], v2['description'],
                                          v2['genres'], v2['themes'], v2['franchises'])
        
        valid = valid1 and valid2

        if not valid:
            y_cos_sims.append(0)
            print("Len of y_cos_sims")
            print(len(y_cos_sims))
            continue
            
        deck = process_text(v2['deck'])
        desc = process_text(v2['description'])
        
        # start recommendation process
        model_sim = model.n_similarity(v1['deck'], v2['deck'])
        if model_sim > model_sim_threshold:
            recs[k2] = model_sim

        model_sim = model.n_similarity(v1['description'], v2['description'])
        if model_sim > model_sim_threshold:
            recs[k2] = model_sim

        for g in v2['genres']:
            if g in v1['genres']:
                model_sim = model.n_similarity(v1['name'], v2['name'])
                recs[k2] = model_sim
        
        for g in v2['themes']:
            if g in v1['themes']:
                model_sim = model.n_similarity(v1['name'], v2['name'])
                recs[k2] = model_sim
        
        for g in v2['franchises']:
            if g in v1['franchises']:
                model_sim = model.n_similarity(v1['name'], v2['name'])
                recs[k2] = model_sim

        if model_sim > 0:
            y_cos_sims.append(model_sim)
        else:
            y_cos_sims.append(0)

        print("Len of y_cos_sims")
        print(len(y_cos_sims))
    
    pdb.set_trace()
    print(y_cos_sims)
    print(len(y_cos_sims))
    print(recs)
    pdb.set_trace()
    total_pairs += calculate_confusion_matrix(y_cos_sims, thresholds, y_true)

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

print("end time: ")
print(time.time() - start_time)

print("final pdb")
pdb.set_trace()