"""
Create proof of concept based on input from class
Input: games like X, where X is some mandatory, solicited recommendation
Data: pretrained Word2Vec model
    word embeddings for X
    word embeddings for each game
    (name, deck, description)
Output: game "similar" to X (based on meeting the float threshold for N word embedding attributes)

Plan is to make a "recommendation" with word embeddings approach
"""

import requests_cache
import pdb
import json
import os
from dotenv import load_dotenv
import time
from nltk.tokenize import word_tokenize
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
import numpy as np

import matplotlib.pyplot as plt

start_time = time.time()
load_dotenv()
session = requests_cache.CachedSession("pretrained_embeddings_cache")

# http://www.gamespot.com/api/games/?api_key=<YOUR_API_KEY_HERE>
GAMESPOT_API_KEY = os.getenv('GAMESPOT_API_KEY')
HEADERS = {'User-Agent': 'Video Game Recommender Comprehensive Project'}
GIANTBOMB_API_KEY = os.getenv('GIANTBOMB_API_KEY')

# input game X
# QUERY = "klonoa door to phantomile"
# QUERY = "super mario rpg"
# QUERY = "vampire the masquerade" - "Alone in the Dark"
# QUERY = "sid meier civilization iv" - gives reasonable answers (i.e., Sid Meier / Civ games)
#QUERY = "final fantasy" # - # gives reasonable answers (i.e., final fantasy series games)
# QUERY = "baldur's gate"
QUERY = "super mario bros"

# https://www.giantbomb.com/api/documentation/#toc-0-17
search_api_url = "https://www.giantbomb.com/api/search/?api_key=" + GIANTBOMB_API_KEY + \
    "&format=json&query=" + QUERY + \
    "&resources=game" + \
    "&resource_type=game" 
    # resources = game details that should inform the results, while resource_type = game recommendation itself
search_game_resp = session.get(search_api_url, headers=HEADERS)
# pdb.set_trace()

search_json = json.loads(search_game_resp.text)
game_results = None

# pdb.set_trace()

# ensure that deck and description exist for cosine similarity step
# check number of page results and grab the first entry that has necessary info (non-null values)

num_results = search_json['number_of_page_results']
game_not_found = True

for i in range(min(num_results, 10)):
    if search_json['results'][i]['deck'] != None and search_json['results'][i]['description'] != None \
    and game_not_found:
        game_results = search_json['results'][i]
        game_not_found = False


if game_results == None:
    print("Input query game not found in API database")
    exit()

# FIXME make sure the code fits the diagram

query_name = game_results['name']
game_deck = game_results['deck']
game_desc = game_results['description']

game_platforms = game_results['platforms']

pdb.set_trace()

GUID = game_results['guid']

pdb.set_trace()

tokenizer = RegexpTokenizer(r'\w+')
stops = set(stopwords.words("english"))

query_deck_data = list(set(tokenizer.tokenize(game_deck.lower())) - stops)
query_desc_data = list(set(tokenizer.tokenize(game_desc.lower())) - stops)
query_desc_data = [desc for desc in query_desc_data if desc.isalpha()]

platform_list = []
for plat in game_platforms:
    for k, v in plat.items():
        if k == 'name':
            platform_list.append(v)

# pdb.set_trace()

# get aspects of GUID - genres, themes, franchises

game_api_url = "https://www.giantbomb.com/api/game/" + \
GUID + "/?api_key=" + GIANTBOMB_API_KEY + \
    "&format=json"
game_api_resp = session.get(game_api_url, headers=HEADERS)

game_api_json = json.loads(game_api_resp.text)
game_api_results = game_api_json['results']

# ground truth for similar_games
similar_games_to_query = game_api_json['results']['similar_games']
# pdb.set_trace()

similar_game_names = []
for i in range(len(similar_games_to_query)):
    similar_game_names.append(similar_games_to_query[i]['name'])

pdb.set_trace()

game_genres = game_api_json['results']['genres']
genre_list = []
for i in range(len(game_genres)):
    genre_list.append(game_genres[i]['name'])

game_themes = game_api_json['results']['themes']
theme_list = []
for i in range(len(game_themes)):
    theme_list.append(game_themes[i]['name'])

game_franchises = game_api_json['results']['franchises']
franchise_list = []
for i in range(len(game_franchises)):
    franchise_list.append(game_franchises[i]['name'])

pdb.set_trace()

#pdb.set_trace()

query_dict = {}
query_dict[query_name] = {'name': query_name,
                         'deck': query_deck_data,
                         'description': query_desc_data,
                         'platforms': game_platforms,
                         'genres': genre_list,
                         'franchises': franchise_list,
                         'themes': theme_list}

pdb.set_trace()

#pdb.set_trace()


# 1. Imports and setup to get reviews from API
"""
get_game_info
Provide dictionary-based information about individual games and their qualities (description, themes, etc.)
Arguments:
            api_key (string): API key for GameSpot
            headers (string): specify User-Agent field
            offset (int): how many pages API should skip before sending new content in response
Returns: game_data (dict): key=name and value=
    id, name, deck, description, genres, themes, franchises, release date, image
"""
def get_game_info(api_key, headers, offset):
    game_url = "http://www.gamespot.com/api/games/?api_key=" + api_key + "&format=json" + \
    "&offset=" + str(offset)
    game_call = session.get(game_url, headers=headers) 
    game_json = json.loads(game_call.text)['results']
    # pdb.set_trace()

    game_data = {}
    for game in game_json:
        game_data[game['name']] = {'id': game['id'], 
                             'name': game['name'],
                             'deck': game['deck'],
                             'description': game['description'], 
                             'genres': game['genres'], 
                             'themes': game['themes'], 
                             'franchises': game['franchises'], 
                             'release_date': game['release_date'],
                             'image': game['image']}
    return game_data

def get_games(api_key, headers, game_count=10):
    # https://stackoverflow.com/questions/38987/how-do-i-merge-two-dictionaries-in-a-single-expression-in-python
    games = {}
    for i in range(game_count):
        new_games = get_game_info(api_key=api_key, headers=headers, offset=i*99)
        games = {**games, **new_games}
        
    return games

games_dict = get_games(api_key=GAMESPOT_API_KEY, headers=HEADERS, game_count=20)
games_test_dict = get_games(api_key=GAMESPOT_API_KEY, headers=HEADERS, game_count=10)
#pdb.set_trace()

"""
2. Perform word embedding step
Use a pretrained model
Compare game X to every game in the dataset
If the cosine similarity is greater than the threshold, return the game
"""

# https://radimrehurek.com/gensim/models/keyedvectors.html
# https://radimrehurek.com/gensim/auto_examples/tutorials/run_wmd.html

# query_dict: query name, deck, description
# go through each game in list and see which is best

deck_data = list( \
    list(set(tokenizer.tokenize(v['deck'].lower()))-stops) \
    for v in games_dict.values() if tokenizer.tokenize(v['deck']) != [])

desc_data = list( \
    list(set(tokenizer.tokenize(v['description'].lower()))-stops) \
    for v in games_dict.values() if tokenizer.tokenize(v['description']) != [])

#pdb.set_trace()

# data: use the pretrained model
import gensim.downloader
model = gensim.downloader.load('glove-wiki-gigaword-50')
#pdb.set_trace()

sims = {}
for k, v in games_dict.items():

    # check name
    if query_name == v['name'] or query_name in v['name']:
        model_similarity = model.n_similarity(query_name, v['name'])
        if model_similarity >= 0.8:
            sims[k] = model_similarity

    deck = list(set(tokenizer.tokenize(v['deck'])) - stops)
    desc = list(set(tokenizer.tokenize(v['description'])) - stops) 
    deck = [d.lower() for d in deck]
    desc = [d.lower() for d in desc if d.isalpha()]

    this_genre = []
    this_theme = []
    this_franchise = []

    genres = v['genres']

    for genre in genres:
        for k1, v1 in genre.items():
            this_genre.append(v1)

    themes = v['themes']

    for theme in themes:
        for k1, v1 in theme.items():
            this_theme.append(v1)

    franchises = v['franchises']

    for franchise in franchises:
        for k1, v1 in franchise.items():
            this_franchise.append(v1)

    # pdb.set_trace()

    #pdb.set_trace()
    
    # check the cosine similarity between the tokenized descriptions to get related games

    min_deck_tokens = min(len(query_deck_data), len(deck))
    min_desc_tokens = min(len(query_desc_data), len(desc))

    min_deck_tokens = max(min_deck_tokens, 5)
    min_desc_tokens = max(min_desc_tokens, 5)

   #  pdb.set_trace()

    if len(query_desc_data) >= min_deck_tokens and len(deck) >= min_deck_tokens:
        model_similarity = model.n_similarity(query_deck_data[0:min_deck_tokens], deck[0:min_deck_tokens])

        if model_similarity >= 0.8:
            sims[k] = model_similarity

    if len(query_desc_data) >= min_desc_tokens and len(desc) >= min_desc_tokens:
        model_similarity = model.n_similarity(query_desc_data[0:min_desc_tokens], desc[0:min_desc_tokens])
        
        if model_similarity >= 0.8:
            sims[k] = model_similarity

    # use genres, themes, and franchises
    if len(genre_list) > 0 and len(this_genre) > 0:

        for g in this_genre:
            if g in genre_list:
                model_similarity = model.n_similarity(genre_list[0], g)
                if model_similarity >= 0.8:
                    sims[k] = model_similarity
                

    if len(theme_list) > 0 and len(this_theme) > 0:
        for g in this_theme:
            if g in theme_list:
                model_similarity = model.n_similarity(theme_list[0], g)
                if model_similarity >= 0.8:
                    sims[k] = model_similarity

    if len(franchise_list) > 0 and len(this_franchise) > 0:
        for g in this_franchise:
            if g in franchise_list:
                model_similarity = model.n_similarity(franchise_list[0], g)
                if model_similarity >= 0.8:
                    sims[k] = model_similarity

    #pdb.set_trace()

    # pdb.set_trace()

print(sims)
max_similiarity = max(sims.values())

print("look for similar games manually")
pdb.set_trace()

topX = 5
count = 0
for k, v in sims.items():
    if v == max_similiarity and count < topX:
        print(k, v)
        count += 1

# calculate precision and recall
# precision - correctly recommended items / total recommended items
# recall - correctly recommended items / total useful recommended items
# Let "useful" recommended items be any item that has the same genre, theme, or franchise

# PR definition
# https://www.sciencedirect.com/science/article/pii/S1110866515000341#b0435

ground_truth_recs = [i for i in similar_game_names if i in games_dict.keys()]
recommender_results = [i for i in similar_game_names if i in sims]  
total_recs = [i for i in sims.keys()]

# precision = correct recs in ground truth
# recall = correct recs / total recs
precision = min(len(recommender_results) / len(ground_truth_recs), 1)
recall = min(len(recommender_results) / len(total_recs), 1)

print("precision, recall")
print(precision)
print(recall)

# TODO try ROC curve to see what happens as precision/recall tradeoff is made

recs = list(sims.keys())

tp_items = [i for i in recs if i in ground_truth_recs]
fp_items = [i for i in recs if i not in ground_truth_recs]
fn_items = [i for i in list(games_dict.keys()) if i not in recs and i in ground_truth_recs]
tn_items = [i for i in list(games_dict.keys()) if i not in recs and i not in ground_truth_recs]
        
fpr = len(fp_items) / (len(fp_items) + len(tn_items))
tpr = len(tp_items) / (len(tp_items) + len(fn_items))

# pdb.set_trace()

fpr_vals = [fpr]
tpr_vals = [tpr]

plt.plot(fpr, tpr)
plt.show()

for i in range(1, 4):
    games_test_dict = get_games(api_key=GAMESPOT_API_KEY, headers=HEADERS, game_count=20 + 10 * i)

    sims = {}
    for k, v in games_test_dict.items():

    # check name
        if query_name == v['name'] or query_name in v['name']:
            model_similarity = model.n_similarity(query_name, v['name'])
            if model_similarity >= 0.8:
                sims[k] = model_similarity

        deck = list(set(tokenizer.tokenize(v['deck'])) - stops)
        desc = list(set(tokenizer.tokenize(v['description'])) - stops) 
        deck = [d.lower() for d in deck]
        desc = [d.lower() for d in desc if d.isalpha()]

        this_genre = []
        this_theme = []
        this_franchise = []

        genres = v['genres']

        for genre in genres:
            for k1, v1 in genre.items():
                this_genre.append(v1)

        themes = v['themes']

        for theme in themes:
            for k1, v1 in theme.items():
                this_theme.append(v1)

        franchises = v['franchises']

        for franchise in franchises:
            for k1, v1 in franchise.items():
                this_franchise.append(v1)

        # pdb.set_trace()

        #pdb.set_trace()
        
        # check the cosine similarity between the tokenized descriptions to get related games

        min_deck_tokens = min(len(query_deck_data), len(deck))
        min_desc_tokens = min(len(query_desc_data), len(desc))

        min_deck_tokens = max(min_deck_tokens, 5)
        min_desc_tokens = max(min_desc_tokens, 5)

    #  pdb.set_trace()

        if len(query_desc_data) >= min_deck_tokens and len(deck) >= min_deck_tokens:
            model_similarity = model.n_similarity(query_deck_data[0:min_deck_tokens], deck[0:min_deck_tokens])

            if model_similarity >= 0.8:
                sims[k] = model_similarity

        if len(query_desc_data) >= min_desc_tokens and len(desc) >= min_desc_tokens:
            model_similarity = model.n_similarity(query_desc_data[0:min_desc_tokens], desc[0:min_desc_tokens])
            
            if model_similarity >= 0.8:
                sims[k] = model_similarity

        # use genres, themes, and franchises
        if len(genre_list) > 0 and len(this_genre) > 0:

            for g in this_genre:
                if g in genre_list:
                    model_similarity = model.n_similarity(genre_list[0], g)
                    if model_similarity >= 0.8:
                        sims[k] = model_similarity
                    

        if len(theme_list) > 0 and len(this_theme) > 0:
            for g in this_theme:
                if g in theme_list:
                    model_similarity = model.n_similarity(theme_list[0], g)
                    if model_similarity >= 0.8:
                        sims[k] = model_similarity

        if len(franchise_list) > 0 and len(this_franchise) > 0:
            for g in this_franchise:
                if g in franchise_list:
                    model_similarity = model.n_similarity(franchise_list[0], g)
                    if model_similarity >= 0.8:
                        sims[k] = model_similarity

        #pdb.set_trace()

        # pdb.set_trace()

    print(sims)
    max_similiarity = max(sims.values())

    print("look for similar games manually")
    # pdb.set_trace()

    topX = 5
    count = 0
    for k, v in sims.items():
        if v == max_similiarity and count < topX:
            print(k, v)
            count += 1

    # calculate precision and recall
    # precision - correctly recommended items / total recommended items
    # recall - correctly recommended items / total useful recommended items
    # Let "useful" recommended items be any item that has the same genre, theme, or franchise

    # PR definition
    # https://www.sciencedirect.com/science/article/pii/S1110866515000341#b0435

    ground_truth_recs = [i for i in similar_game_names if i in games_dict.keys()]
    recommender_results = [i for i in similar_game_names if i in sims]  
    total_recs = [i for i in sims.keys()]

    # precision = correct recs in ground truth
    # recall = correct recs / total recs
    precision = min(len(recommender_results) / len(ground_truth_recs), 1)
    recall = min(len(recommender_results) / len(total_recs), 1)

    print("precision, recall")
    print(precision)
    print(recall)

    # TODO try ROC curve to see what happens as precision/recall tradeoff is made

    recs = list(sims.keys())

    tp_items = [i for i in recs if i in ground_truth_recs]
    fp_items = [i for i in recs if i not in ground_truth_recs]
    fn_items = [i for i in list(games_dict.keys()) if i not in recs and i in ground_truth_recs]
    tn_items = [i for i in list(games_dict.keys()) if i not in recs and i not in ground_truth_recs]
            
    # https://stackoverflow.com/questions/41757653/how-to-compute-aucarea-under-curve-for-recommendation-system-evaluation
    fpr = len(fp_items) / (len(fp_items) + len(tn_items))
    tpr = len(tp_items) / (len(tp_items) + len(fn_items))

    pdb.set_trace()

    fpr_vals.append(fpr)
    tpr_vals.append(tpr)

# before, didn't have evaluation metrics.
# now, use supervised learning to say that similar_games are ground_truth and use PR based on that
# currently have PR and ROC curve
# TODO
# 1. optimize for the metric. reduce FPR (false positive rate) and keep TPR high
# 2. reduce total recommendations so that entire universe isn't recommended
# 3. continue algorithm to improve performance
# other metrics not necessarily needed

plt.plot(fpr_vals, tpr_vals)
plt.show()
# pdb.set_trace()