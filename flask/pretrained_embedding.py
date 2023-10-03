"""
Create proof of concept based on input from class
Input: games like X, where X is some mandatory, solicited recommendation
Data: pretrained Word2Vec model
    word embeddings for X
    word embeddings for each game
    (name, deck, description)
Output: game "similar" to X

Evaluation metric: 
Use GiantBomb "similar games" to reverse engineer recommendation process
This transforms recommendation into a supervised learning project
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

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import sklearn.metrics as metrics

import matplotlib.pyplot as plt
import random

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
# QUERY = "vampire the masquerade"
# QUERY = "sid meier civilization iv"
#QUERY = "final fantasy"
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

for i in range(min(num_results, 5)):
    if search_json['results'][i]['deck'] != None and search_json['results'][i]['description'] != None \
    and game_not_found:
        game_results = search_json['results'][i]
        game_not_found = False


if game_results == None:
    print("Input query game not found in API database")
    exit()

query_name = game_results['name']
game_deck = game_results['deck']
game_desc = game_results['description']

game_platforms = game_results['platforms']

#pdb.set_trace()

GUID = game_results['guid']

#pdb.set_trace()

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
print("similar_games_to_query")
# pdb.set_trace()

similar_game_names = []
game_count = 5
similar_threshold = 5
similar_threshold = min(len(similar_games_to_query), 5)

for i in range(min(len(similar_games_to_query), similar_threshold)):
# for i in range(len(similar_games_to_query)):
    name = similar_games_to_query[i]['name']
    guid_val = similar_games_to_query[i]['api_detail_url'][35:-1] # check pdb for confirmation
    similar_game_names.append({name: guid_val})
    #pdb.set_trace()

# use api_detail_url in similar_games[i] to call game API on each similar game
# then, store genres, franchises etc
# append dict to games_dict
# we have GUID from 
print("similar games set trace")
# pdb.set_trace()

# get ground truth games to avoid division by zero error
# proof of concept - get 3, or as many as there are, whichever is less
# get all info about these games and add them to games_dict to ensure presence of ground truth

# do proof of concept

if len(similar_game_names) == 0:
    print("no similar games found")
    exit()

sample_game_len = min(len(similar_game_names), similar_threshold)
# sample_game_len = len(similar_game_names)

print(sample_game_len)
#pdb.set_trace()

sample_similar_games = similar_game_names[0:sample_game_len]
#pdb.set_trace()

def get_game_demographics(json_file, dict_key):

    # check to make sure genre, theme, franchise, etc. is in the game results json
    results_json = json_file['results']
    if dict_key not in results_json.keys():
        return ['']
    elif json_file['results'][dict_key] == None:
        return ['']
    elif json_file['results'][dict_key] == []:
        return ['']
    
    # else, the appropriate call can be made from the API
    
    call = json_file['results'][dict_key]
    call_list = []
    for i in range(len(call)):
        call_list.append(call[i]['name'])

    return call_list

similar_games_dict = {} # initialize here so that ground truth games are guaranteed

print("check sg & sample_similar_games")
#pdb.set_trace()

for sg in sample_similar_games:
    for k, v in sg.items(): # key = name and value = GUID
        # https://www.giantbomb.com/api/documentation/#toc-0-17
        search_sample_url = "https://www.giantbomb.com/api/game/" + \
            v + "/?api_key=" + GIANTBOMB_API_KEY + \
            "&format=json"
        sample_resp = session.get(search_sample_url, headers=HEADERS)
        # pdb.set_trace()

        # search_json = json.loads(search_game_resp.text)
        sample_json = json.loads(sample_resp.text)
        sample_results = sample_json['results']

        print("sample_results for: " + str(k))
        # pdb.set_trace()

        for i in range(min(len(sample_results), 1)):
            name = sample_results['name']
            deck = sample_results['deck']
            desc = sample_results['description']
            platforms = sample_results['platforms']


            deck_data = list(set(tokenizer.tokenize(deck.lower())) - stops)
            desc_data = list(set(tokenizer.tokenize(desc.lower())) - stops)
            desc_data = [desc for desc in desc_data if desc.isalpha()]

            deck_str = ''
            for d in deck_data:
                entry = d + " "
                deck_str += entry
            
            desc_str = ''
            for d in desc_data:
                entry = d + " "
                desc_str += entry

            print("check string deck and description")
            # pdb.set_trace()

            genre_list = get_game_demographics(sample_json, 'genres')
            theme_list = get_game_demographics(sample_json, 'themes')
            franchise_list = get_game_demographics(sample_json, 'franchises')

            similar_games_dict[name] = {'name': name,
                         'deck': deck,
                         'description': desc,
                         'platforms': platforms,
                         'genres': genre_list,
                         'franchises': franchise_list,
                         'themes': theme_list,
                         'recommended': 1} # used in y_true

            # print("check ground truth query_dict")
            # pdb.set_trace()


print("check ground truth games")
#pdb.set_trace()

genre_list = get_game_demographics(game_api_json, 'genres')
theme_list = get_game_demographics(game_api_json, 'themes')
franchise_list = get_game_demographics(game_api_json, 'franchises')

#pdb.set_trace()

#pdb.set_trace()

query_dict = {}
query_dict[query_name] = {'name': query_name,
                         'deck': query_deck_data,
                         'description': query_desc_data,
                         'platforms': game_platforms,
                         'genres': genre_list,
                         'franchises': franchise_list,
                         'themes': theme_list}

print("check ground truth games & dataset games")
# pdb.set_trace()

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
                             'image': game['image'],
                             'recommended' : 0} # used in y_true
    return game_data

def get_games(api_key, headers, game_count=10, loop_offset=0):
    # https://stackoverflow.com/questions/38987/how-do-i-merge-two-dictionaries-in-a-single-expression-in-python
    games = {}
    for i in range(game_count):
        new_offset = i*99 + loop_offset
        new_games = get_game_info(api_key=api_key, headers=headers, offset=new_offset)
        games = {**games, **new_games}
        
    return [games, new_offset]

games = get_games(api_key=GAMESPOT_API_KEY, headers=HEADERS, game_count=5, loop_offset=0)
dataset_games_dict = games[0]
offset = games[1]

# games_test_dict = get_games(api_key=GAMESPOT_API_KEY, headers=HEADERS, game_count=20)
#pdb.set_trace()

"""
2. Perform word embedding step
Use a pretrained model
Compare game X to every game in the dataset
If the cosine similarity is greater than the threshold, return the game
"""

#pdb.set_trace()

# data: use the pretrained model
import gensim.downloader
model = gensim.downloader.load('glove-wiki-gigaword-50')
#pdb.set_trace()

sims = {}
fpr_list = []
tpr_list = []

# put similar_games GiantBomb API results into dataset games from GameSpot API
total_games_dict = {** dataset_games_dict, ** similar_games_dict}

# randomly shuffle dictionary keys to mix ground truth games with games_dict
# https://stackoverflow.com/questions/19895028/randomly-shuffling-a-dictionary-in-python
temp_list = list(total_games_dict.items())
random.shuffle(temp_list)
total_games_dict = dict(temp_list)

#pdb.set_trace()

y_true = [v['recommended'] for k, v in total_games_dict.items()]


print("fix games_dict shuffle")
#pdb.set_trace()

model_sim_threshold = 0.2

    # games_dict = get_games(api_key=GAMESPOT_API_KEY, headers=HEADERS, game_count=30, loop_offset=0 + 2 * i)

    # https://radimrehurek.com/gensim/models/keyedvectors.html
    # https://radimrehurek.com/gensim/auto_examples/tutorials/run_wmd.html

    # query_dict: query name, deck, description
    # go through each game in list and see which is best

# Can't use ground truth inside algorithm guts. Instead, 
# use definition of ROC curve.
"""
Start with thresholds - uniform list of floats
0.1, 0.2, 0.3, ..., 0.9
Max granularity is 1 / len(dataset)
Then, from threshold, we get TP and FP (generated from threshold)
Get those ordered pairs
When we specify a TP and FP, do reverse lookup to get the threshold
That threshold is what's used as cos similarity cutoff threshold
Deliverable for demo: ROC curve handcrafted using this idea
"""

thresholds = list(np.linspace(0.1, 1, 10)) # [0.1, 0.2, 0.3, ..., 1]
thresholds = [round(i, 2) for i in thresholds]

y_cos_sim = []
for k, v in total_games_dict.items():
    model_sim_threshold = 0.9
    model_sim = 0

    deck_data = list( \
    list(set(tokenizer.tokenize(v['deck'].lower()))-stops) \
    for v in total_games_dict.values() if tokenizer.tokenize(v['deck']) != [])

    desc_data = list( \
    list(set(tokenizer.tokenize(v['description'].lower()))-stops) \
    for v in total_games_dict.values() if tokenizer.tokenize(v['description']) != [])

    # check name
    if query_name == v['name'] or query_name in v['name']:
        model_sim = model.n_similarity(query_name, v['name'])
        if model_sim >= model_sim_threshold:
            sims[k] = model_sim

    deck = list(set(tokenizer.tokenize(v['deck'])) - stops)
    desc = list(set(tokenizer.tokenize(v['description'])) - stops) 
    deck = [d.lower() for d in deck]
    desc = [d.lower() for d in desc if d.isalpha()]

    this_genre = []
    this_theme = []
    this_franchise = []

    genres = v['genres']

    for genre in genres:

        if isinstance(genre, str):
            this_genre.append(genre)
        elif isinstance(genre, dict):
            
            for k1, v1 in genre.items():
                this_genre.append(v1)

    themes = v['themes']

    for theme in themes:
        if isinstance(theme, str):
            this_genre.append(theme)
        elif isinstance(theme, dict):
            
            for k1, v1 in theme.items():
                this_theme.append(v1)

    franchises = v['franchises']

    for franchise in franchises:
        if isinstance(franchise, str):
            this_genre.append(franchise)
        elif isinstance(franchise, dict):
            
            for k1, v1 in franchise.items():
                this_franchise.append(v1)

    # pdb.set_trace()

    #pdb.set_trace()

    # check the cosine similarity between the tokenized descriptions to get related games

    # FIXME model_sim_threshold is obtained from ROC
    # see photo from pictures/notes

    min_deck_tokens = min(len(query_deck_data), len(deck))
    min_desc_tokens = min(len(query_desc_data), len(desc))

    min_deck_tokens = max(min_deck_tokens, 10)
    min_desc_tokens = max(min_desc_tokens, 10)

    #  pdb.set_trace()

    if len(query_desc_data) >= min_deck_tokens and len(deck) >= min_deck_tokens:
        model_sim = model.n_similarity(query_deck_data[0:min_deck_tokens], deck[0:min_deck_tokens])

        if model_sim >=  model_sim_threshold:
            sims[k] = model_sim

    if len(query_desc_data) >= min_desc_tokens and len(desc) >= min_desc_tokens:
        model_sim = model.n_similarity(query_desc_data[0:min_desc_tokens], desc[0:min_desc_tokens])
        
        if model_sim >=  model_sim_threshold:
            sims[k] = model_sim

    # use genres, themes, and franchises
    if len(genre_list) > 0 and len(this_genre) > 0:

        for g in this_genre:
            if g in genre_list:
                model_sim = model.n_similarity(genre_list[0], g)
                if model_sim >=  model_sim_threshold:
                    sims[k] = model_sim
                

    if len(theme_list) > 0 and len(this_theme) > 0:
        for g in this_theme:
            if g in theme_list:
                model_sim = model.n_similarity(theme_list[0], g)
                if model_sim >=  model_sim_threshold:
                    sims[k] = model_sim

    if len(franchise_list) > 0 and len(this_franchise) > 0:
        for g in this_franchise:
            if g in franchise_list:
                model_sim = model.n_similarity(franchise_list[0], g)
                if model_sim >=  model_sim_threshold:
                    sims[k] = model_sim
    
    if model_sim > 0: # added cosine similarity to rec
        y_cos_sim.append(model_sim)
    else:
        y_cos_sim.append(0) # not recommending

    print(y_cos_sim)
            

# calculate TPR and FPR using thresholds
fpr = []
tpr = []

fp_tp_pairs = []

# https://stats.stackexchange.com/questions/123124/how-to-determine-the-optimal-threshold-for-a-classifier-and-generate-roc-curve
# https://stackoverflow.com/questions/61321778/how-to-calculate-tpr-and-fpr-in-python-without-using-sklearn
# https://stackoverflow.com/questions/2951701/is-it-possible-to-use-else-in-a-list-comprehension
# https://stackoverflow.com/questions/477486/how-do-i-use-a-decimal-step-value-for-range
# https://stackoverflow.com/questions/4843173/how-to-check-if-type-of-a-variable-is-string

reverse_lookup = {}

for t in thresholds:

    # use cosine similarity as probability
    y_pred = [1 if i > t else 0 for i in y_cos_sim]
    #tp = [i for i in y_pred if i == 1 and i == 1]
    #fp = [i for i in y_pred if y_pred[i] == 1 and y_true[i] == 0]
    #fn = [i for i in y_pred if y_pred[i] == 0 and y_true[i] == 1]
    #tn = [i for i in y_pred if y_pred[i] == 0 and y_true[i] == 0]

    tp = 0
    fp = 0
    fn = 0
    tn = 0

    for i in range(len(y_pred)):
        if y_pred[i] == 1 and y_true[i] == 1:
            tp += 1
        elif y_pred[i] == 1 and y_true[i] == 0:
            fp += 1
        elif y_pred[i] == 0 and y_true[i] == 1:
            fn += 1
        elif y_pred[i] == 0 and y_true[i] == 0:
            tn += 1



    # pdb.set_trace()

    tpr = tp / (tp + fn)
    fpr = fp / (fp + tn)

    print((fpr, tpr))
    fp_tp_pairs.append((fpr, tpr))

    # save this fpr and tpr pair corresponding threshold value
    # do this for reverse lookup table to pick a specific threshold from TPR, FPR vals

    reverse_lookup[t] = (fpr, tpr)


pdb.set_trace()
print(reverse_lookup)

fvals = []
tvals = []
for ft in fp_tp_pairs:
    fvals.append(ft[0])
    tvals.append(ft[1])

lin_x = np.linspace(0.0, 1.0, 11)
lin_y = np.linspace(0.0, 1.0, 11)

plt.plot(fvals, tvals)
plt.plot(lin_x, lin_y, label='linear')  # Plot some data on the (implicit) axes.
plt.xlabel("FPR")
plt.ylabel("TPR")
plt.title("ROC curve")
plt.show()

print(fp_tp_pairs)

print("sims")
print(sims)
pdb.set_trace()
max_sim = max(sims.values())

topX = 5
count = 0
for k, v in sims.items():
    if v == max_sim and count < topX:
        # print(k, v)
        count += 1

print("y_pred")
print(y_pred[0:10])
print("y_true")
print(y_true[0:10])
pdb.set_trace()

"""
Now, if we want to choose a specific point on the curve,
just re-run a new trial using that threshold (access TPR and FPR based on reverse_lookup)
"""

# pdb.set_trace()
#print("sims so far")
#print(sims)
#pdb.set_trace()
#games = get_games(api_key=GAMESPOT_API_KEY, headers=HEADERS, game_count=2, loop_offset=offset)
#games_dict = games[0]
#offset = games[1]

end_time = time.time() - start_time
print("seconds: ", end_time)