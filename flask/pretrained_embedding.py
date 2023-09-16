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

start_time = time.time()
load_dotenv()
session = requests_cache.CachedSession("pretrained_embeddings_cache")

# http://www.gamespot.com/api/games/?api_key=<YOUR_API_KEY_HERE>
GAMESPOT_API_KEY = os.getenv('GAMESPOT_API_KEY')
HEADERS = {'User-Agent': 'Video Game Recommender Comprehensive Project'}
GIANTBOMB_API_KEY = os.getenv('GIANTBOMB_API_KEY')

# input game X
# QUERY = "klonoa door to phantomile"
QUERY = "super mario rpg"
# QUERY = "vampire the masquerade"
# QUERY = "sid meier civilization iv"

# https://www.giantbomb.com/api/documentation/#toc-0-17
game_url = "https://www.giantbomb.com/api/search/?api_key=" + GIANTBOMB_API_KEY + \
    "&format=json&query=" + QUERY + \
    "&resources=game" + \
    "&resource_type=game" 
    # resources = game details that should inform the results, while resource_type = game recommendation itself
game_resp = session.get(game_url, headers=HEADERS)
# pdb.set_trace()

game_json = json.loads(game_resp.text)
game_results = None

pdb.set_trace()

# ensure that deck and description exist for cosine similarity step
# check number of page results and grab the first entry that has necessary info (non-null values)

num_results = game_json['number_of_page_results']
game_not_found = True

for i in range(min(num_results, 10)):
    if game_json['results'][i]['deck'] != None and game_json['results'][i]['description'] != None \
    and game_not_found:
        game_results = game_json['results'][i]
        game_not_found = False


if game_results == None:
    print("Input query game not found in API database")
    exit()

# FIXME make sure the code fits the diagram

game_name = game_results['name']
game_deck = game_results['deck']
game_desc = game_results['description']

tokenizer = RegexpTokenizer(r'\w+')
stops = set(stopwords.words("english"))

query_deck_data = list(set(tokenizer.tokenize(game_deck.lower())) - stops)
query_desc_data = list(set(tokenizer.tokenize(game_desc.lower())) - stops)
query_desc_data = [desc for desc in query_desc_data if desc.isalpha()]

#pdb.set_trace()

query_dict = {}
query_dict[game_name] = {'name': game_name,
                         'deck': query_deck_data,
                         'description': query_desc_data}

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

games_dict = get_games(api_key=GAMESPOT_API_KEY, headers=HEADERS, game_count=3)
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
    deck = list(set(tokenizer.tokenize(v['deck'])) - stops)
    desc = list(set(tokenizer.tokenize(v['description'])) - stops) 
    deck = [d.lower() for d in deck]
    desc = [d.lower() for d in desc if d.isalpha()]

    #pdb.set_trace()
    
    # check the cosine similarity between the tokenized descriptions to get related games

    if len(query_desc_data) >= 5 and len(deck) >= 5:
        model_similarity = model_similarity = model.n_similarity(query_deck_data[0:5], deck[0:5])
        sims[k] = model_similarity

    if len(query_desc_data) >= 5 and len(desc) >= 5:
        model_similarity = model.n_similarity(query_desc_data[0:5], desc[0:5])
        sims[k] = model_similarity
    #pdb.set_trace()

    # pdb.set_trace()

print(sims)
max_similiarity = max(sims.values())

print("look for similar games manually")
pdb.set_trace()

for k, v in sims.items():
    if v == max_similiarity:
        print(k, v)

"""
sentence_obama = 'Obama speaks to the media in Illinois'.lower().split()

sentence_president = 'The president greets the press in Chicago'.lower().split()
>>>

similarity = word_vectors.wmdistance(sentence_obama, sentence_president)

print(f"{similarity:.4f}")
3.4893
>>>

distance = word_vectors.distance("media", "media")

print(f"{distance:.1f}")
0.0
>>>

similarity = word_vectors.n_similarity(['sushi', 'shop'], ['japanese', 'restaurant'])

print(f"{similarity:.4f}")
0.7067

"""

#import gensim.downloader
#model = gensim.downloader.load('glove-wiki-gigaword-50')
#pdb.set_trace()