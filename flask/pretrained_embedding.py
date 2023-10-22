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

from gensim.test.utils import common_texts
from gensim.corpora.dictionary import Dictionary

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
#QUERY = "vampire the masquerade"
#QUERY = "sid meier civilization"
#QUERY = "ultima"
#QUERY = "team fortress 2"
#QUERY = "final fantasy"
#QUERY = "baldur's gate"
#QUERY = "super mario bros"
#QUERY = "pokemon red"

# TODO automate some games - try to get several games from dataset automatically rather than handpicked
# maybe get game titles from csv? or API? either way try to automate instead of handpicking
# https://www.kaggle.com/datasets/rush4ratio/video-game-sales-with-ratings ???

QUERY_GAMES = ["fez", "limbo","red dead redemption", "witcher 2", "zelda wind waker", "kerbal space program",
              "sonic the hedgehog", "faster than light", "oregon trail", "thomas was alone", "jet set radio",
              "the legend of zelda", "space invaders", "klonoa door to phantomile", "super mario bros", "pokemon red", "baldur's gate", "team fortress 2", "vampire the masquerade",
              "super mario rpg", "ultima", "worms 2", "pacman", "galaga", "space invaders", 
              "mario kart", "castlevania symphony of the night", "donkey kong country",
              "command and conquer", "civilization 5", "metal gear solid"]

games_used = 0

# data: use the pretrained model
import gensim.downloader
model = gensim.downloader.load('glove-wiki-gigaword-50')
print("check model")
#pdb.set_trace()

total_fprs_tprs = []

total_reverse_lookup = []
total_tpr_lookup = []
total_fpr_lookup = []

sims_total_values = []
total_cosine_sims = []

# 1. Imports and setup to get reviews from API
"""
get_game_info
Provide dictionary-based information about individual games and their qualities (description, themes, etc.)
Gets reviews and foreign keys into the game
Arguments:
            api_key (string): API key for GameSpot
            headers (string): specify User-Agent field
            offset (int): how many pages API should skip before sending new content in response
Returns: game_data (dict): key=name and value=
    id, name, deck, description, genres, themes, franchises, release date, image
"""
def get_game_info(api_key, headers, offset):

    # try reviews
    #rev_url = "http://www.gamespot.com/api/reviews/?api_key=" + api_key + "&format=json"
    #rev_call = session.get(rev_url, headers=headers)
    #rev_json = json.loads(rev_call.text)
    #print("rev trace")
    # pdb.set_trace()
    #pdb.set_trace()

    game_data = {}

    game_url = "http://www.gamespot.com/api/games/?api_key=" + api_key + "&format=json" + \
        "&offset=" + str(offset)
    game_call = session.get(game_url, headers=headers) 
    game_json = json.loads(game_call.text)['results']

    #for rev in rev_json['results']:
        #game_api_url = rev['game']['api_detail_url'] + "&api_key=" + api_key + "&format=json"
        #game_call = session.get(game_api_url, headers=headers)
        #game_json = json.loads(game_call.text)

        #print("include review")
        #pdb.set_trace()

    for game in game_json:
        
        print("include game")
        #pdb.set_trace()

        """
        game_url = "http://www.gamespot.com/api/games/?api_key=" + api_key + "&format=json" + \
        "&offset=" + str(offset)
        game_call = session.get(game_url, headers=headers) 
        game_json = json.loads(game_call.text)['results']
        print("game info pdb set trace")
        pdb.set_trace()

        game_data = {}
        for game in game_json:

            review_foreign_key = game['reviews_api_url'][45:]
            review_url = "http://www.gamespot.com/api/reviews/?api_key=" + api_key + "&format=json" + \
            "&association:" + str(game['id'])
            #"&" + review_foreign_key
            print(review_url)
            #pdb.set_trace()

            review_call = session.get(review_url, headers=headers)
            review_json = json.loads(review_call.text)
            print("check review fk from gamespot")
            pdb.set_trace()

            #print(review_json['results'])

            #pdb.set_trace()

            # review_url = "http://www.gamespot.com/api/reviews/?api_key=" + api_key + "&format=json" + \
        """

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

def get_games(api_key, headers, game_count=10):
    # https://stackoverflow.com/questions/38987/how-do-i-merge-two-dictionaries-in-a-single-expression-in-python
    games = {}
    for i in range(game_count):
        new_offset = i*99
        new_games = get_game_info(api_key=api_key, headers=headers, offset=new_offset)
        games = {**games, **new_games}
        
    return [games, new_offset]

games = get_games(api_key=GAMESPOT_API_KEY, headers=HEADERS, game_count=2)
dataset_games_dict = games[0]
offset = games[1]

print("games called for dataset")

for QUERY in QUERY_GAMES:

    print("CURRENTLY ON " + QUERY)

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
        continue

    #pdb.set_trace()


    query_name = game_results['name']
    game_deck = game_results['deck']
    game_desc = game_results['description']
    game_platforms = game_results['platforms']

    #pdb.set_trace()

    GUID = game_results['guid']

    #pdb.set_trace()

    # try to get GameSpot review for query game

    #query_rev_url = "http://www.gamespot.com/api/games/?api_key=" + GAMESPOT_API_KEY + \
    #   "&format=json&filter=name:" + query_name
    #query_rev_call = session.get(query_rev_url, headers=HEADERS) 
    #query_json = json.loads(query_rev_call.text)['results']

    #print("check query review GameSpot")
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

    print("game_api_results")
    #pdb.set_trace()

    if 'genres' in game_api_results and game_api_results['genres'] != None:
        try:
            query_genre = game_api_results['genres'][0]['name']
        except TypeError:
            query_genre = ''
    else:
        query_genre = ''
    
    if 'themes' in game_api_results and game_api_results['themes'] != None:
        try:
            query_theme = game_api_results['themes'][0]['name']
        except TypeError:
            query_theme = ''
    else:
        query_theme = ''

    if 'franchises' in game_api_results and game_api_results['franchises'] != None:
        try:
            query_franchise = game_api_results['franchises'][0]['name']
        except TypeError:
            query_franchise = ''
    else:
        query_franchise = ''

    # ground truth for similar_games
    similar_games_to_query = game_api_json['results']['similar_games']
    print("similar_games_to_query")
    # pdb.set_trace()

    sample_similar_games = []
    game_count = 5
    #similar_threshold = 5
    #similar_threshold = len(similar_games_to_query) #min(len(similar_games_to_query), 5)

    similar_found = True

    if similar_games_to_query == None:
        sample_similar_games = []
        similar_found = False
    else:
        for i in range(len(similar_games_to_query)):
        # for i in range(len(similar_games_to_query)):
            name = similar_games_to_query[i]['name']
            guid_val = similar_games_to_query[i]['api_detail_url'][35:-1] # check pdb for confirmation
            sample_similar_games.append({name: guid_val})
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

        if len(sample_similar_games) == 0 or similar_found == False:
            print("no similar games found")
            continue

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

    if sample_similar_games == None or len(sample_similar_games) == 0:
        continue 

    for sg in sample_similar_games:
        for k, v in sg.items(): # key = name and value = GUID
            # https://www.giantbomb.com/api/documentation/#toc-0-17
            search_sample_url = "https://www.giantbomb.com/api/game/" + \
                v + "/?api_key=" + GIANTBOMB_API_KEY + \
                "&format=json"
            sample_resp = session.get(search_sample_url, headers=HEADERS)
            # pdb.set_trace()

            # search_json = json.loads(search_game_resp.text)
            #print("check json response text")
            # pdb.set_trace()

            if sample_resp == None or sample_resp.text == None:
                continue

            sample_json = json.loads(sample_resp.text)
            sample_results = sample_json['results']

            print("sample_results for: " + str(k))
            # pdb.set_trace()

            for i in range(min(len(sample_results), 1)):
                name = sample_results['name']

                if sample_results['deck'] != None:
                    deck = sample_results['deck']
                    deck_data = list(set(tokenizer.tokenize(deck.lower())) - stops)
                else:
                    deck = ''
                    deck_data = ''

                if sample_results['description'] != None:
                    desc = sample_results['description']
                    desc_data = list(set(tokenizer.tokenize(desc.lower())) - stops)
                    desc_data = [desc for desc in desc_data if desc.isalpha()]
                else:
                    desc = ''
                    desc_data = ''

                print("check string deck and description")
                # pdb.set_trace()

                genre_list = get_game_demographics(sample_json, 'genres')
                theme_list = get_game_demographics(sample_json, 'themes')
                franchise_list = get_game_demographics(sample_json, 'franchises')

                similar_games_dict[name] = {'name': name,
                            'deck': deck,
                            'description': desc,
                            #'platforms': platforms,
                            'genres': genre_list,
                            'franchises': franchise_list,
                            'themes': theme_list,
                            'review': None,
                            'recommended': 1} # used in y_true

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

    # games_test_dict = get_games(api_key=GAMESPOT_API_KEY, headers=HEADERS, game_count=20)
    #pdb.set_trace()

    """
    2. Perform word embedding step
    Use a pretrained model
    Compare game X to every game in the dataset
    If the cosine similarity is greater than the threshold, return the game
    """

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
    thresholds = list(np.linspace(0.1, 1, 100))
    thresholds = [round(i, 2) for i in thresholds]

    # reasons_dict may double count some specific entries
    reasons_dict = {
        'name': 0,
        'deck': 0,
        'desc': 0,
        'genre': 0,
        'franchise': 0,
        'theme': 0,
        'rev': 0
    }

    y_cos_sim = []

    # recommendation algorithm

    model_sim_threshold = 0.9
    for k, v in total_games_dict.items():
        
        model_sim = 0
        no_nulls = True

        deck = list(set(tokenizer.tokenize(v['deck'])) - stops)
        desc = list(set(tokenizer.tokenize(v['description'])) - stops) 
        
        deck = [d.lower() for d in deck]
        desc = [d.lower() for d in desc if d.isalpha() and len(d) > 1]

        min_deck_tokens = min(len(query_deck_data), len(deck))
        min_desc_tokens = min(len(query_desc_data), len(desc))

        min_deck_tokens = max(min_deck_tokens, 1)
        min_desc_tokens = max(min_desc_tokens, 1)

        # if any attribute (deck, description, genre, theme, franchise) is null, don't recommend

        game_attributes = [v['name'], v['deck'], v['description'], v['genres'], v['themes'], v['franchises']]

        for g in game_attributes:
            if g == None or g == '':
                model_sim = 0
                no_nulls = False

        if len(query_deck_data) >= min_deck_tokens and len(deck) >= min_deck_tokens and no_nulls:
            model_sim = model.n_similarity(query_deck_data, deck)
            if model_sim >= model_sim_threshold:
                sims[k] = model_sim
                reasons_dict['deck'] += 1
            
        if len(query_desc_data) >= min_desc_tokens and len(desc) >= min_desc_tokens and no_nulls:
            model_sim = model.n_similarity(query_desc_data, desc)
            if model_sim >= model_sim_threshold:
                sims[k] = model_sim
                reasons_dict['desc'] += 1


        # if genre, franchise, or theme adds a game,
        # add the cosine similarity of the query name and the recommendation name

        if no_nulls:
            for g in v['genres']:

                if isinstance(g, str):
                    if query_genre == g and query_genre != '':

                        #model_sim = model.n_similarity(query_desc_data, desc)
                        #sims[k] = model_sim
                        
                        model_sim = model.n_similarity(query_name, v['name'])
                        sims[k] = model_sim
                        reasons_dict['genre'] += 1

                elif isinstance(g, dict):
                    if query_genre in g.values() and query_genre != '':
                        
                        #model_sim = model.n_similarity(query_desc_data, desc)
                        #sims[k] = model_sim

                        model_sim = model.n_similarity(query_name, v['name'])
                        sims[k] = model_sim
                        reasons_dict['genre'] += 1
            
            for g in v['themes']:

                if isinstance(g, str):
                    if query_theme == g and query_theme != '':

                        #model_sim = model.n_similarity(query_desc_data, desc)
                        #sims[k] = model_sim

                        model_sim = model.n_similarity(query_name, v['name'])
                        sims[k] = model_sim
                        reasons_dict['theme'] += 1

                elif isinstance(g, dict):
                    if query_theme in g.values() and query_theme != '':
                        
                        #model_sim = model.n_similarity(query_desc_data, desc)
                        #sims[k] = model_sim

                        model_sim = model.n_similarity(query_name, v['name'])
                        sims[k] = model_sim
                        reasons_dict['theme'] += 1
            
            for g in v['franchises']:

                if isinstance(g, str):
                    if query_franchise == g and query_franchise != '':

                        #model_sim = model.n_similarity(query_desc_data, desc)
                        #sims[k] = model_sim
                        
                        model_sim = model.n_similarity(query_name, v['name'])
                        sims[k] = model_sim
                        reasons_dict['franchise'] += 1

                elif isinstance(g, dict):
                    if query_franchise in g.values() and query_franchise != '':
                        
                        #model_sim = model.n_similarity(query_desc_data, desc)
                        #sims[k] = model_sim

                        model_sim = model.n_similarity(query_name, v['name'])
                        sims[k] = model_sim
                        reasons_dict['franchise'] += 1

        if model_sim > 0: # added cosine similarity to rec
            y_cos_sim.append(model_sim)
        else:
            y_cos_sim.append(0) # not recommending

        total_cosine_sims.append(model_sim)

        # print(y_cos_sim)
                
    print("check uniq words")
    #pdb.set_trace()

    print("check reasons for adding to sims")
    print(reasons_dict)
    #pdb.set_trace()

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
    tpr_lookup = {}
    fpr_lookup = {}

    tp_tot = 0
    fp_tot = 0
    fn_tot = 0
    tn_tot = 0

    for t in thresholds:

        # use cosine similarity as probability
        y_pred = [1 if i > t else 0 for i in y_cos_sim]

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

        tp_tot += tp
        fp_tot += fp
        fn_tot += fn
        tn_tot += tn

        # pdb.set_trace()

        tpr = tp / (tp + fn)
        fpr = fp / (fp + tn)

        print((fpr, tpr, t))
        fp_tp_pairs.append((fpr, tpr, t))

        print("Current threshold: ", t)
        print("tp, fp, fn, tn")
        print(tp_tot, fp_tot, fn_tot, tn_tot)

        # save this fpr and tpr pair corresponding threshold value
        # do this for reverse lookup table to pick a specific threshold from TPR, FPR vals

        reverse_lookup[t] = (fpr, tpr, t)
        tpr_lookup[tpr] = {'fpr': fpr, 't': t}
        tpr_lookup[fpr] = {'tpr': tpr, 't': t}


    
    total_fprs_tprs.append(fp_tp_pairs)
    total_reverse_lookup.append(reverse_lookup)
    total_tpr_lookup.append(tpr_lookup)
    total_fpr_lookup.append(fpr_lookup)
    sims_total_values += [v for v in sims.values()]

    print("total")
    print("tp, fp, fn, tn")
    print(tp_tot, fp_tot, fn_tot, tn_tot)

    #pdb.set_trace()
    print(reverse_lookup)

    fvals = []
    tvals = []
    for ft in fp_tp_pairs:
        fvals.append(ft[0])
        tvals.append(ft[1])

    lin_x = np.linspace(0.0, 1.0, 11)
    lin_y = np.linspace(0.0, 1.0, 11)

    #plt.plot(fvals, tvals)
    #plt.plot(lin_x, lin_y, label='linear')  # Plot some data on the (implicit) axes.
    #plt.xlabel("FPR")
    #plt.ylabel("TPR")
    #new_title = "ROC curve: " + QUERY
    #plt.title(new_title)
    #plt.show()

    print(fp_tp_pairs)

    print("sims")
    print(sims)
    #pdb.set_trace()
    max_sim = max(sims.values())

    topX = 5
    count = 0
    for k, v in sims.items():
        if v == max_sim and count < topX:
            print(k, v)
            count += 1

    print("y_pred")
    print(y_pred[0:10])
    print("y_true")
    print(y_true[0:10])
    #pdb.set_trace()

    """
    Now, if we want to choose a specific point on the curve,
    just re-run a new trial using that threshold (access TPR and FPR based on reverse_lookup)
    """

    games_used += 1

    end_time = time.time() - start_time
    # print("seconds: ", end_time)
    print("minutes: ", end_time / 60)

# take the average of several games ROC to see what it looks like
fvals = [0] * len(total_fprs_tprs[0])
tvals = [0] * len(total_fprs_tprs[0])
threshold_vals = [0] * len(total_fprs_tprs[0])

"""
0.1: (1.0, 0.96), 0.11: (1.0, 0.96),
0.12: (1.0, 0.96), 0.13: (1.0, 0.96), 0.14: (1.0, 0.96), 0.15: (1.0, 0.96), 0.16: (1.0, 0.96), 0.17: (1.0, 0.96), 0.18: (1.0, 0.96), 0.19: (1.0, 0.96), 0.2: (1.0, 0.96), 0.21: (1.0, 0.96), 0.22: (1.0, 0.96), 0.23: (1.0, 0.96), 0.24: (1.0, 0.96), 0.25: (1.0, 0.96), 0.26: (1.0, 0.96), 0.27: (1.0, 0.96), 0.28: (1.0, 0.96), 0.29: (1.0, 0.96), 0.3: (1.0, 0.96), 0.31: (1.0, 0.96), 0.32: (1.0, 0.96), 0.33: (1.0, 0.96), 0.34: (1.0, 0.96), 0.35: (1.0, 0.96), 0.36: (1.0, 0.96), 0.37: (1.0, 0.96), 0.38: (1.0, 0.96), 0.39: (1.0, 0.96), 0.4: (1.0, 0.96), 0.41: (1.0, 0.96), 0.42: (1.0, 0.96), 0.43: (1.0, 0.96),
"""
print(total_fprs_tprs)

pdb.set_trace()

# sum all TPR and FPR values on a per game, per tuple basis
for i in range(len(total_fprs_tprs)):
    for j in range(len(total_fprs_tprs[i])):
        #print("j")
        #pdb.set_trace()
        fvals[j] += total_fprs_tprs[i][j][0]
        tvals[j] += total_fprs_tprs[i][j][1]
        threshold_vals[j] += total_fprs_tprs[i][j][2]
        print("threshold ", total_fprs_tprs[i][j][0], total_fprs_tprs[i][j][1], total_fprs_tprs[i][j][2])
        #pdb.set_trace()

pdb.set_trace()

# then take the average
avg_fvals = [fvals[j] / len(total_fprs_tprs) for j in range(len(fvals))]
avg_tvals = [tvals[j] / len(total_fprs_tprs) for j in range(len(tvals))]
avg_thresholds = [threshold_vals[j] / len(threshold_vals) for j in range(len(threshold_vals))]

"""
# also get the reverse lookup values to start finding performance jumps
# check reverse value to see where "jumps" are
print("check total_reverse_lookup")

sum_lookup_fpr = [0] * len(total_reverse_lookup) * len(thresholds)
sum_lookup_tpr = [0] * len(total_reverse_lookup) * len(thresholds)
ct = 0
for i in range(len(total_reverse_lookup)):
    print("iteration " + str(i) + " belonging to " + QUERY_GAMES[i])

    for t in thresholds:
        print(i, total_reverse_lookup[i][t][0], total_reverse_lookup[i][t][1])
        sum_lookup_fpr[ct] = total_reverse_lookup[i][t][0]
        sum_lookup_tpr[ct] = total_reverse_lookup[i][t][1]
        ct += 1

#sum_lookup_fpr = sum_lookup_fpr / (len(total_reverse_lookup) * len(thresholds))
#sum_lookup_tpr = sum_lookup_tpr / (len(total_reverse_lookup) * len(thresholds))
#avg_fpr = 

#avg_fpr = [sum_lookup_fpr[j] / (len(total_reverse_lookup) * len(thresholds)) for j in range(len(sum_lookup_fpr))]
#avg_tpr = [sum_lookup_tpr[j] / (len(total_reverse_lookup) * len(thresholds)) for j in range(len(sum_lookup_tpr))]

print("avg_fpr and tpr")
for i in range(len(fvals)):
    print(i, fvals[i], tvals[i])
#print(avg_fpr)
#print(avg_tpr)
"""

pdb.set_trace()

lin_x = np.linspace(0.0, 1.0, 11)
lin_y = np.linspace(0.0, 1.0, 11)

pdb.set_trace()

plt.plot(avg_fvals, avg_tvals)
plt.plot(lin_x, lin_y, label='linear')  # Plot some data on the (implicit) axes.
plt.xlabel("FPR")
plt.ylabel("TPR")
plt.title("ROC curve, averaged")
plt.show()

print((time.time() - start_time) / 60)
print("minutes")

print("games_used")
print(games_used)

print("use total reverse lookup to check for unusual points")

print("individual games")
for i in range(len(total_reverse_lookup)):
    for t in thresholds[::-1]:
        print(i, t, total_reverse_lookup[i][t])

pdb.set_trace()

print("aggregated ROC points")
for i in range(len(avg_fvals[::-1])):
    print(i, avg_fvals[::-1][i], avg_tvals[::-1][i], avg_thresholds[::-1][i]) # look up threshold from this ordered pair
    #if avg_fvals[::-1][i] in total_fpr_lookup[i].keys():
      #  print("f threshold: " + str(total_fpr_lookup[i][avg_fvals[::-1][i]]))
    #i#f avg_tvals[::-1][i] in total_tpr_lookup[i].keys():
     #   print("t threshold: " + str(total_tpr_lookup[i][avg_tvals[::-1][i]]))

"""
Try generating a plot of histograms
Then research how using embeddings as entry to word encoder might help
(Instead of feeding in raw plaintext strings)
"""

print("check histograms")
pdb.set_trace()

# plot histogram
#counts, bins = np.histogram([v for k, v in sims.items()])
#plt.stairs(counts, bins)
#plt.show()

#counts, bins = np.histogram(sims_total_values)
#plt.stairs(counts, bins)
#plt.show()

plt.hist(sims_total_values, bins=100)
plt.title("total accepted rec cos sims")
plt.show()

#counts, bins = np.histogram(total_cosine_sims)
#plt.stairs(counts, bins)
#plt.show()

plt.hist(total_cosine_sims, bins=100)
plt.title("total rec cos sims")
plt.show()

pdb.set_trace()

print("end time: ")
print(time.time() - start_time)

print("final pdb")
pdb.set_trace()