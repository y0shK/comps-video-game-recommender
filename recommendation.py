"""
Supervised learning project to reverse engineer recommended games from GiantBomb API
"""
import requests_cache
import pdb
import os
from dotenv import load_dotenv
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random

import tensorflow_hub as hub
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

from sklearn.decomposition import PCA
from imblearn.over_sampling import SMOTE
from collections import Counter

from get_api_info import get_giantbomb_game_info, get_gamespot_games, get_similar_games
from process_recs import process_text, check_valid_name, check_valid_deck_and_desc, get_embedding, check_valid_demographics
from visuals import update_demographic_dict, create_histogram

start_time = time.time()
load_dotenv()
session = requests_cache.CachedSession("recommendation_cache")

# http://www.gamespot.com/api/games/?api_key=<YOUR_API_KEY_HERE>
GAMESPOT_API_KEY = os.getenv('GAMESPOT_API_KEY')
HEADERS = {'User-Agent': 'Video Game Recommender Comprehensive Project'}
GIANTBOMB_API_KEY = os.getenv('GIANTBOMB_API_KEY')

# https://www.kaggle.com/datasets/dahlia25/metacritic-video-game-comments - use metacritic_game_info.csv to parse
csv_titles_df = pd.read_csv("metacritic_game_info.csv")

"""
1. Form a query set by combining games from metacritic csv and GameSpot API (all recommendation boolean == 0)
Get titles from each of the data sources, then get information from API calls
"""
csv_titles = list(set([i for i in csv_titles_df['Title']][0:15])) # 10 games
print(csv_titles[0])
print(len(csv_titles))
pdb.set_trace()

query_set = {}
for title in csv_titles:
    query_dict = get_giantbomb_game_info(api_key=GIANTBOMB_API_KEY, query=title, headers=HEADERS,session=session)
    query_set = {**query_set, **query_dict}

pdb.set_trace()

# get gamespot games - (2 * 99) games before filtering
gamespot_games = get_gamespot_games(api_key=GAMESPOT_API_KEY, headers=HEADERS, game_count=2, session=session)

query_set = {**query_set, **gamespot_games}
print("dataset size: ", len(query_set))

"""
2. Generate a train and test set for the model 
"""
# set up tf universal encoder for cosine similarity comparisons on sentence embeddings
module_url = "https://tfhub.dev/google/universal-sentence-encoder/4"
model = hub.load(module_url)
print(type(model))
print("tf universal encoder set up!")

# X is a list of deck and description embeddings, as generated by encoder
# y is the recommendation boolean
X = []
y = []

genres_dict_1 = {}
themes_dict_1 = {}
franchises_dict_1 = {}

genres_dict_0 = {}
themes_dict_0 = {}
franchises_dict_0 = {}

print("Length of dataset:")
print(len(query_set))

game_counter = 1

for k, v in query_set.items():

    # on a per game basis,
    # check similar games (which are recommended == 1 contingent on the query set game)
    # and check all other query set games (which are recommended == 0 since they are not similar)
    similar_games_instance = get_similar_games(api_key=GIANTBOMB_API_KEY, query=k, headers=HEADERS, session=session)
    if similar_games_instance == None or similar_games_instance == {}:
        continue

    # train the SVM on both initial game (k) and potential recommendation (signal: sk or noise: nk)
    # e.g., SVM(initial = Breakout and rec = Tetris) = 1
    # SVM(initial = BG3 and rec = Tetris) = 0
    # use name, deck, and description

    if check_valid_name(v['name']) == False:
        continue

    if check_valid_deck_and_desc(v['deck'], v['description']) == False:
        continue

    init_name = process_text(v['name'])
    init_deck = process_text(v['deck'])
    init_desc = process_text(v['description'])
    init_tokenized_list = init_name + init_deck + init_desc

    for sk, sv in similar_games_instance.items():
        name = sv['name']
        deck = sv['deck']
        desc = sv['description']

        if check_valid_name(name) == False:
            continue

        if check_valid_deck_and_desc(deck, desc) == False:
            continue

        tokenized_name = process_text(name)
        tokenized_deck = process_text(deck)
        tokenized_desc = process_text(desc)
        tokenized_list = init_tokenized_list + tokenized_name + tokenized_deck + tokenized_desc
        word_embedding = get_embedding(model, tokenized_list)

        X.append(word_embedding)
        y.append(1)

        # add demographics (genre, theme, franchise) to see the frequency of each appearance with rec=1
        known_genre = check_valid_demographics(sv['genres'])
        known_theme = check_valid_demographics(sv['themes'])
        known_franchise = check_valid_demographics(sv['franchises'])

        if known_genre:
            current_genre = sv['genres'][0]
        else:
            current_genre = "unknown"

        if known_theme:
            current_theme = sv['themes'][0]
        else:
            current_theme = "unknown"
        
        if known_franchise:
            current_franchise = sv['franchises'][0]
        else:
            current_franchise = "unknown"

        genres_dict_1 = update_demographic_dict(current_genre, genres_dict_1)
        themes_dict_1 = update_demographic_dict(current_theme, themes_dict_1)
        franchises_dict_1 = update_demographic_dict(current_franchise, franchises_dict_1)

    for nk, nv in query_set.items():
        if nk in similar_games_instance:
            continue

        name = nv['name']
        deck = nv['deck']
        desc = nv['description']
        not_similar_embed = {}

        if check_valid_name(name) == False:
            continue

        if check_valid_deck_and_desc(deck, desc) == False:
            continue

        known_genre = check_valid_demographics(nv['genres'])
        known_theme = check_valid_demographics(nv['themes'])
        known_franchise = check_valid_demographics(nv['franchises'])

        tokenized_name = process_text(name)
        tokenized_deck = process_text(deck)
        tokenized_desc = process_text(desc)
        tokenized_list = init_tokenized_list + tokenized_name + tokenized_deck + tokenized_desc
        word_embedding = get_embedding(model, tokenized_list)

        X.append(word_embedding)
        y.append(0)

        # add genre and theme to see the frequency of each appearance with rec=1
        known_genre = check_valid_demographics(nv['genres'])
        known_theme = check_valid_demographics(nv['themes'])
        known_franchise = check_valid_demographics(nv['franchises'])

        if known_genre:
            current_genre = nv['genres'][0]
        else:
            current_genre = "unknown"

        if known_theme:
            current_theme = nv['themes'][0]
        else:
            current_theme = "unknown"

        if known_franchise:
            current_franchise = nv['franchises'][0]
        else:
            current_franchise = "unknown"

        genres_dict_0 = update_demographic_dict(current_genre, genres_dict_0)
        themes_dict_0 = update_demographic_dict(current_theme, themes_dict_0)
        franchises_dict_0 = update_demographic_dict(current_franchise, franchises_dict_0)

    print("currently on iteration", game_counter)
    game_counter += 1

print("check dataset X and y")
pdb.set_trace()

print("Original dataset shape")
print(len(X))
print(len(y))
print(Counter(y))

pdb.set_trace()

sm = SMOTE(random_state=42)
X_res, y_res = sm.fit_resample(X, y)

print("Examine X and y of resample")
pdb.set_trace()
print(len(X_res))
print(len(y_res))

print("Resampled dataset shape")
pdb.set_trace()
print(Counter(y_res))

# shuffle around X_res and y_res
tuple_list = []

for i in range(len(y_res)):
    tuple_list.append((X_res[i], y_res[i]))

random.shuffle(tuple_list)
X_res_shuffle = [-1] * len(tuple_list)
y_res_shuffle = [-1] * len(tuple_list)

for i in range(len(tuple_list)):
    X_res_shuffle[i] = tuple_list[i][0]
    y_res_shuffle[i] = tuple_list[i][1]

print("check shuffle")
pdb.set_trace()

X_res = X_res_shuffle
y_res = y_res_shuffle

# train test split - 80% train, 20% test
split = int(0.8 * len(X_res))
X_train = np.array(X_res[0:split])
y_train = np.array(y_res[0:split])
X_test = np.array(X_res[split:])
y_test = np.array(y_res[split:])

print("check train test split")
pdb.set_trace()

samples = [X_train, X_test, y_train, y_test]
sample_strings = ['X_train', 'X_test', 'y_train', 'y_test']
for sample in samples:
    unique, counts = np.unique(sample, return_counts=True)
    print("", sample, " ", unique, ": ", counts)

"""
3a. Perform dimensionality reduction with PCA to project the N-dimensional word embedding vectors onto R^2
This will make the vectors easier to visualize to assess SVM performance

3b. Feed 2-D train and test set into SVM and evaluate the model.
"""
clf = make_pipeline(StandardScaler(), SVC(gamma='auto', kernel='rbf'))
# https://stackoverflow.com/questions/71386142/valueerror-x-has-2-features-but-minmaxscaler-is-expecting-1-features-as-input
pca = PCA(n_components = 2) # we want 2 dimensions to visualize

X_train_lowdim = pca.fit_transform(X_train)
X_test_lowdim = pca.fit_transform(X_test)

print("X_train_lowdim: ")
print(X_train_lowdim[0:2])
print(y_train[0:2])

print("X_test_lowdim length")
print(len(X_test_lowdim))

clf.fit(X_train_lowdim, y_train) 
y_preds_lowdim = clf.predict(X_test_lowdim)

print("y_preds")
unique, counts = np.unique(y_preds_lowdim, return_counts=True)
print("y_preds: ", unique, counts)

print("length of y_preds_lowdim, and y_test")
print(len(y_preds_lowdim))
print(len(y_test))

# evaluations
print("F-1 score: ", f1_score(y_test, y_preds_lowdim, average='binary'))
cm = confusion_matrix(y_test, y_preds_lowdim)
print(cm)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.show()

# add some test cases for evaluation
# fit PCA on test1_query and a test1_query similar game
# then transform PCA on test1_rec and test1_rec2
# this is: SVM(query = Breakout, rec = Tetris) -> 1
# SVM(query = Breakout, rec = Baldur's Gate) -> 0
# to test, get information of query and proposed recommendation from API
# then tokenize to get into right format and feed into SVM
test1_query = 'Breakout'
test1_rec = 'Tetris'
test1_rec2 = "Baldur's Gate"

# fit PCA on test1_query and similar game
sim_game = get_similar_games(api_key=GIANTBOMB_API_KEY, query=test1_query, headers=HEADERS, max_similar=1, session=session)

for k, v in sim_game.items():
    sim_game_name = v['name']
    sim_game_deck = v['deck']
    sim_game_desc = v['description']

sim_tokenized_list = process_text(sim_game_name + sim_game_deck + sim_game_desc)
sim_embedding = get_embedding(model, sim_tokenized_list)

test1_query_info = get_giantbomb_game_info(api_key=GIANTBOMB_API_KEY, query=test1_query, headers=HEADERS, session=session)
test1_rec_info = get_giantbomb_game_info(api_key=GIANTBOMB_API_KEY, query=test1_rec, headers=HEADERS, session=session)
test1_rec2_info = get_giantbomb_game_info(api_key=GIANTBOMB_API_KEY, query=test1_rec2, headers=HEADERS, session=session)

print("check game info")
pdb.set_trace()

for k, v in test1_query_info.items():
    test1_query_name = v['name']

for k, v in test1_rec_info.items():
    test1_rec_name = v['name']

for k, v in test1_rec2_info.items():
    test1_rec2_name = v['name']

t1qname = test1_query_info[test1_query_name]['name']
t1qdeck = test1_query_info[test1_query_name]['deck']
t1qdesc = test1_query_info[test1_query_name]['description']

t1rname = test1_rec_info[test1_rec_name]['name']
t1rdeck = test1_rec_info[test1_rec_name]['deck']
t1rdesc = test1_rec_info[test1_rec_name]['description']

t1rname2 = test1_rec2_info[test1_rec2_name]['name']
t1rdeck2 = test1_rec2_info[test1_rec2_name]['deck']
t1rdesc2 = test1_rec2_info[test1_rec2_name]['description']

print("check extracted attributes")
pdb.set_trace()

query_tokenized_list = process_text(t1qname + t1qdeck + t1qdesc)
rec_tokenized_list = process_text(t1rname + t1rdeck + t1rdesc)
rec2_tokenized_list = process_text(t1rname2 + t1rdeck2 + t1rdesc2)

query_embed = get_embedding(model, query_tokenized_list)
rec1_embed = get_embedding(model, rec_tokenized_list) 
rec2_embed = get_embedding(model, rec2_tokenized_list)

query_sim_fit_lowdim = pca.fit([query_embed] + [sim_embedding])

rec1_lowdim = pca.transform([rec1_embed])
rec1_pred = clf.predict(rec1_lowdim)
print('Given', test1_query, 'recommend', test1_rec, ':', rec1_pred)

rec2_lowdim = pca.transform([rec2_embed])
rec2_pred = clf.predict(rec2_lowdim)
print('Given', test1_query, 'recommend', test1_rec2, ':', rec1_pred)

pdb.set_trace()

print("check evaluations")
pdb.set_trace()

# show double bar chart of game demographics
create_histogram("Genres", genres_dict_0, genres_dict_1, 3)
create_histogram("Themes", themes_dict_0, themes_dict_1, 3)
create_histogram("Franchises", franchises_dict_0, franchises_dict_1, 3)

print("check visualizations")
pdb.set_trace()

print("final pdb")
pdb.set_trace()

print("final time")
finaltime = time.time() - start_time
print("final time (min): ", finaltime/60)
