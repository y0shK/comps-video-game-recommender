"""
Write wrappers to get information from API to clean up recommendation script
"""
import requests_cache
import json
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
import pdb

my_session = requests_cache.CachedSession("my_new_cache")
tokenizer = RegexpTokenizer(r'\w+')
stops = set(stopwords.words("english"))

"""
get_gamespot_game_info
Provide dictionary-based information about individual games and their qualities (description, themes, etc.)
Arguments:
    api_key (string): API key for GameSpot
    headers (string): specify User-Agent field
    offset (int): how many pages API should skip before sending new content in response
    session (CachedSession): optional session to store results in local cache
Returns: game_data (dict): key=name and value=
    id, name, deck, description, genres, themes, franchises
"""
def get_gamespot_game_info(api_key, headers, offset, session=my_session):
    game_data = {}
    game_url = "http://www.gamespot.com/api/games/?api_key=" + api_key + "&format=json" + \
        "&offset=" + str(offset)
    game_call = session.get(game_url, headers=headers) 

    try:
        game_json = json.loads(game_call.text)['results']
    except ValueError:
        return {}

    # normalize qualities to get same data structure as GiantBomb games
    for game in game_json:
        if 'genres' in game.keys() and len(game['genres']) >= 1:
            genre_list = normalize_qualities(game['genres'])
        else:
            genre_list = ['']

        if 'themes' in game.keys() and len(game['themes']) >= 1:
            theme_list = normalize_qualities(game['themes'])
        else:
            theme_list = ['']

        if 'franchises' in game.keys() and len(game['franchises']) >= 1:
            franchise_list = normalize_qualities(game['franchises'])
        else:
            franchise_list = ['']

        game_data[game['name']] = {'id': game['id'], 
                            'name': game['name'],
                            'deck': game['deck'],
                            'description': game['description'], 
                            'genres': genre_list, 
                            'themes': theme_list, 
                            'franchises': franchise_list,
                            'review': '',
                            'body': '',
                            'lede': ''}
    return game_data

"""
get_gamespot_games
Use get_gamespot_game_info() as a helper function to return dataset of games
Arguments: 
    api_key (string): API key for GameSpot
    headers (string): specify User-Agent field
    game_count (int): how many pages desired (roughly 100 games per page)
Returns:
    games (dict): {k: game name, v: {game properties}}
"""
def get_gamespot_games(api_key, headers, game_count=10, session=my_session):
    # https://stackoverflow.com/questions/38987/how-do-i-merge-two-dictionaries-in-a-single-expression-in-python
    games = {}
    for i in range(game_count):
        new_offset = i*99
        new_games = get_gamespot_game_info(api_key=api_key, headers=headers, offset=new_offset, session=session)
        games = {**games, **new_games} 
    return games

"""
get_game_demographics
Helper function for get_giantbomb_game_info()
Gets genre, theme, and franchise programmatically, with error checks in case of null json content
Arguments:
    json_file (json)
    dict_key (string): "genres", "themes", etc. Can expand to other parts of json if needed
Returns:
    call_list (string): list of content inside each json category
Sample run:
    theme_list = get_game_demographics(sample_json, 'themes')
    theme_list -> ["fantasy", "adventure"]
"""
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

"""
normalize_qualities
Normalize genres, themes, and franchises to all be of format [''] or contain content
Can take GiantBomb or GameSpot API
Arguments:
    quality (list): genres, themes, franchises, etc.
Returns:
    quality_list (list): a list of qualities, each of format [''] so that string entries are filtered out
"""
def normalize_qualities(quality):
    q_list = []
    if isinstance(quality, str): # giantbomb
        q_list.append(quality)

    elif isinstance(quality, list) and quality[0] and isinstance(quality[0], dict): # gamespot
        for qi in range(len(quality)):
            for k, v in quality[qi].items():
                q_list.append(v)

    elif isinstance(quality, list) and not quality[0]: # gamespot
        q_list.append([''])

    elif isinstance(quality, list): # giantbomb
        q_list += quality

    else:
        q_list.append([''])

    return q_list

"""
get_giantbomb_game_info
Get information from GiantBomb API for a given query game (e.g., coming from the csv read-in)
Arguments:
    api_key (string): API key for GiantBomb
    query (string): query to search within API call
    headers (string): specify User-Agent field
    session (CachedSession): optional session to store results in local cache
Returns: 
    result (dict): {k: game name, v: {game properties}} (or dummy game in null cases)
"""
def get_giantbomb_game_info(api_key, query, headers, session=my_session):
    # https://www.giantbomb.com/api/documentation/#toc-0-17
    search_api_url = "https://www.giantbomb.com/api/search/?api_key=" + api_key + \
        "&format=json&query=" + query + \
        "&resources=game" + \
        "&resource_type=game" 
        # resources = game details that should inform the results, while resource_type = game recommendation itself
    search_game_resp = session.get(search_api_url, headers=headers)

    try:
        search_json = json.loads(search_game_resp.text)
    except ValueError:
        return {}
    
    game_results = None
    num_results = search_json['number_of_page_results']
    game_not_found = True
    dummy_game = {'': {
        'id': '', 
        'name': '',
        'deck': '',
        'description': '', 
        'genres': '', 
        'themes': '', 
        'franchises': '',
        'review': '',
        'body': '',
        'lede': ''
    }}

    for i in range(min(num_results, 5)):
        if search_json['results'][i]['deck'] != None and search_json['results'][i]['description'] != None \
        and game_not_found:
            game_results = search_json['results'][i]
            game_not_found = False

    if game_results == None or game_not_found:
        return dummy_game

    name = game_results['name']
    deck = game_results['deck']
    desc = game_results['description']
    GUID = game_results['guid']

    # preprocess deck and description
    if deck == None:
        deck = ''
    if desc == None:
        desc = ''

    # get aspects of GUID - genres, themes, franchises
    game_api_url = "https://www.giantbomb.com/api/game/" + \
    GUID + "/?api_key=" + api_key + \
        "&format=json"
    game_api_resp = session.get(game_api_url, headers=headers)

    try:
        game_api_json = json.loads(game_api_resp.text)
    except ValueError:
        return {}
    game_api_results = game_api_json['results']

    # set unexpected input to empty string, and manually set cosine similarity to 0 later to account for it
    if 'genres' in game_api_results and game_api_results['genres'] != None:
        try:
            query_genre = normalize_qualities(game_api_results['genres'][0]['name'])
        except TypeError:
            query_genre = ['']
    else:
        query_genre = ['']
    
    if 'themes' in game_api_results and game_api_results['themes'] != None:
        try:
            query_theme = normalize_qualities(game_api_results['themes'][0]['name'])
        except TypeError:
            query_theme = ['']
    else:
        query_theme = ['']

    if 'franchises' in game_api_results and game_api_results['franchises'] != None:
        try:
            query_franchise = normalize_qualities(game_api_results['franchises'][0]['name'])
        except TypeError:
            query_franchise = ['']
    else:
        query_franchise = ['']

    # develop game dict to return
    query_game_dict = {}
    query_game_dict[name] = { 
        'name': name,
        'deck': deck,
        'description': desc, 
        'genres': query_genre, 
        'themes': query_theme, 
        'franchises': query_franchise, 
        'review': '',
        'body': '',
        'lede': '',
        'guid': GUID # provide in returned dict for external use via future API calls
    }
    return query_game_dict

"""
get_gamespot_associated_review_games
Add reviews from GameSpot to the dataset
Get reviews from GameSpot games, then find those games on GiantBomb to add demographic information
Arguments: 
    gamespot_key (string): API key for GameSpot
    giantbomb_key (string): API key for GiantBomb
    headers (string): specify User-Agent field
    session (CachedSession): optional session to store results in local cache
Returns:
    review_data (dict): key=name and value=
    id, name, deck, description, genres, themes, franchises
"""
def get_gamespot_associated_review_games(gamespot_key, giantbomb_key, headers, session):
    review_url = "http://www.gamespot.com/api/reviews/" + "?api_key=" + gamespot_key + "&format=json"
    review_resp = session.get(review_url, headers=headers)

    try:
        review_json = json.loads(review_resp.text)
    except ValueError:
        return {}
    
    results = review_json['results']
    games = {}
    for rev in results:
        if rev['game']['name'] != "" and rev['body'] != "" and rev['lede'] != "":
            name = rev['game']['name']
            body = rev['body']
            lede = rev['lede']
            games[name] = {'body': body, 'lede': lede}

    associated_review_dict = {}
    for k, v in games.items():
        demographic = get_giantbomb_game_info(api_key=giantbomb_key, query=k, headers=headers, session=session)
        add_to_dict = True

        if demographic:
            if k in demographic:
                for kd, vd in demographic[k].items():
                    if vd == '':
                        add_to_dict = False
                    
                if add_to_dict:
                    demographic[k]['body'] = v['body']
                    demographic[k]['lede'] = v['lede']
                    associated_review_dict[k] = demographic

    return associated_review_dict

"""
get_similar_games
Given an input query game g (coming from csv or GameSpot), 
provide the recommended games for g (as determined by GiantBomb API)
Arguments:
    api_key (string): API key for GameSpot
    query (string): query to search within API call
    headers (string): specify User-Agent field
    max_similar (int): optional argument to artificially cap similar game amount
    session (CachedSession): optional session to store results in local cache
Returns:
    similar_games (dict): {k: game name, v: {game properties}},
    or return {} in null case
"""
def get_similar_games(api_key, query, headers, max_similar=-1, session=my_session):
    # search to get api response for given query
    # if not found, return {}
    # else, get the similar games for the query game
    # then get a dictionary with all of their properties
    # return that dictionary

    # get api response for given query
    search_api_url = "https://www.giantbomb.com/api/search/?api_key=" + api_key + \
        "&format=json&query=" + query + \
        "&resources=game" + \
        "&resource_type=game" 
    search_game_resp = session.get(search_api_url, headers=headers)

    if search_game_resp == None or search_game_resp == '':
        return {}

    # https://stackoverflow.com/questions/62609264/how-to-catch-json-decoder-jsondecodeerror
    try:
        search_json = json.loads(search_game_resp.text)
    except ValueError:
        return {}
    game_results = None

    num_results = search_json['number_of_page_results']
    game_not_found = True

    # return null entry if no query similar game is found
    for i in range(min(num_results, 5)):
        if search_json['results'][i]['deck'] != None and search_json['results'][i]['description'] != None \
        and game_not_found:
            game_results = search_json['results'][i]
            game_not_found = False

    if game_results == None or game_not_found:
        return {}
    
    # else, get all the similar games for the given query
    # first get the game object itself
    GUID = game_results['guid']
    game_api_url = "https://www.giantbomb.com/api/game/" + \
    GUID + "/?api_key=" + api_key + \
        "&format=json"
    game_api_resp = session.get(game_api_url, headers=headers)

    try:
        game_api_json = json.loads(game_api_resp.text)
    except ValueError:
        return {}

    game_api_results = game_api_json['results']
    # then find similar games using the game object results
    similar_games_to_query = game_api_results['similar_games']
    similar_games_list = []

    if similar_games_to_query == None:
        # return dummy game 
        # no similar games worth noting
        return {}
    
    if max_similar == -1: # no user argument provided
        max_similar = len(similar_games_to_query) # don't artificially cap

    for i in range(min(len(similar_games_to_query), max_similar)):
        name = similar_games_to_query[i]['name']
        guid_val = similar_games_to_query[i]['api_detail_url'][35:-1]
        similar_games_list.append({name: guid_val})

    similar_games_output = {}
    for sg in similar_games_list:
        for k, v in sg.items():
            # call API to get information for game
            # append information to dictionary
            # add to dataset such that game ought to be recommended (boolean == 1)
            search_sample_url = "https://www.giantbomb.com/api/game/" + \
                v + "/?api_key=" + api_key + \
                "&format=json"
            
            sample_resp = session.get(search_sample_url, headers=headers)
            if sample_resp == None or sample_resp.text == None or sample_resp.text == "":
                continue

            if sample_resp.text != None:
                try:
                    sample_json = json.loads(sample_resp.text)
                    sample_results = sample_json['results']
                except ValueError:
                    continue
            else:
                continue

            for i in range(min(len(sample_results), 1)):
                if sample_results['name'] != None:
                    name = sample_results['name']
                else:
                    name = ''

                if sample_results['deck'] != None:
                    deck = sample_results['deck']
                else:
                    deck = ''

                if sample_results['description'] != None:
                    desc = sample_results['description']
                else:
                    desc = ''

                genre_list = get_game_demographics(sample_json, 'genres')
                theme_list = get_game_demographics(sample_json, 'themes')
                franchise_list = get_game_demographics(sample_json, 'franchises')

                similar_games_output[name] = {'name': name,
                            'deck': deck,
                            'description': desc,
                            'genres': genre_list,
                            'franchises': franchise_list,
                            'themes': theme_list,
                            'review': '',
                            'body': '',
                            'lede': ''}
    return similar_games_output