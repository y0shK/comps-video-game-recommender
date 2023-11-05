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
from sklearn.metrics import RocCurveDisplay

from sklearn.decomposition import PCA
from mlxtend.plotting import plot_decision_regions
from imblearn.over_sampling import SMOTE
from collections import Counter

from get_api_info import get_giantbomb_game_info, get_gamespot_games, get_similar_games
from process_recs import process_text, check_valid_deck_and_desc, get_embedding, check_valid_demographics
from visuals import update_demographic_dict, create_histogram

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

game_counter = 1
for k, v in query_set.items():

    # on a per game basis,
    # check similar games (which are recommended == 1 contingent on the query set game)
    # and check all other query set games (which are recommended == 0 since they are not similar)
    similar_games_instance = get_similar_games(api_key=GIANTBOMB_API_KEY, query=k, headers=HEADERS, session=session)
    if similar_games_instance == None or similar_games_instance == {}:
        continue
    
    for sk, sv in similar_games_instance.items():
        deck = sv['deck']
        desc = sv['description']

        if check_valid_deck_and_desc(deck, desc) == False:
            continue

        tokenized_deck = process_text(deck)
        tokenized_desc = process_text(desc)
        tokenized_list = tokenized_deck + tokenized_desc
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

        deck = nv['deck']
        desc = nv['description']

        if check_valid_deck_and_desc(deck, desc) == False:
            continue

        known_genre = check_valid_demographics(nv['genres'])
        known_theme = check_valid_demographics(nv['themes'])
        known_franchise = check_valid_demographics(nv['franchises'])

        tokenized_deck = process_text(deck)
        tokenized_desc = process_text(desc)
        tokenized_list = tokenized_deck + tokenized_desc
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
    
    print("currently on game", game_counter, "of", len(query_set))
    game_counter += 1

print("check dataset X and y")
pdb.set_trace()

print("Original dataset shape")
print(Counter(y))

pdb.set_trace()

sm = SMOTE(random_state=42)
X_res, y_res = sm.fit_resample(X, y)

print("Examine genres and themes of resample")
pdb.set_trace()

print("Resampled dataset shape")
pdb.set_trace()
print(Counter(y_res))

print("Examine SMOTE")
print(X_res[0:1])
print(y_res[0:10])
pdb.set_trace()

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

clf.fit(X_train_lowdim, y_train) 
y_preds_lowdim = clf.predict(X_test_lowdim)

print("y_preds")
unique, counts = np.unique(y_preds_lowdim, return_counts=True)
print("y_preds: ", unique, counts)

# evaluations
print("F-1 score: ", f1_score(y_test, y_preds_lowdim, average='binary'))
cm = confusion_matrix(y_test, y_preds_lowdim)
print(cm)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.show()

RocCurveDisplay.from_estimator(clf, X_test_lowdim, y_test)
lin_x = np.linspace(0.0, 1.0, 11)
lin_y = np.linspace(0.0, 1.0, 11)
plt.plot(lin_x, lin_y, label='linear')  # Plot some data on the (implicit) axes.
plt.xlabel("FPR")
plt.ylabel("TPR")
plt.title("ROC curve")
plt.show()

# show double bar chart of game demographics
create_histogram("Genres", genres_dict_0, genres_dict_1, 3)
create_histogram("Themes", themes_dict_0, themes_dict_1, 3)
create_histogram("Franchises", franchises_dict_0, franchises_dict_1, 3)

print("check evaluations")
pdb.set_trace()

print("plot decision regions")
pdb.set_trace()

X_lowdim = pca.fit_transform(X_res)
clf.fit(X_lowdim, y_res)
plot_decision_regions(np.array(X_lowdim), np.array(y_res), clf=clf, legend=2)

# FIXME clean up code
# FIXME clean up git repo - unused files

# Adding axes annotations
plt.xlabel('Word embedding value (dense)')
plt.ylabel('Game recommendation')
plt.title('Support Vector Machine visualization')
plt.xlim(-1, 1)
plt.ylim(-1, 1)
plt.show()

print("check visualizations")
pdb.set_trace()

print("final pdb")
pdb.set_trace()

print("final time")
finaltime = time.time() - start_time
print("final time (min): ", finaltime/60)
