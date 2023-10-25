import requests_cache
import json
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords

my_session = requests_cache.CachedSession("my_new_cache")

tokenizer = RegexpTokenizer(r'\w+')
stops = set(stopwords.words("english"))

"""
get_gamespot_game_info
Provide dictionary-based information about individual games and their qualities (description, themes, etc.)
Gets reviews and foreign keys into the game
Arguments:
    api_key (string): API key for GameSpot
    headers (string): specify User-Agent field
    offset (int): how many pages API should skip before sending new content in response
    session (CachedSession): optional session to store results in local cache
Returns: game_data (dict): key=name and value=
    id, name, deck, description, genres, themes, franchises, recommend_boolean
"""
def get_gamespot_game_info(api_key, headers, offset, session=my_session):

    game_data = {}
    game_url = "http://www.gamespot.com/api/games/?api_key=" + api_key + "&format=json" + \
        "&offset=" + str(offset)
    game_call = session.get(game_url, headers=headers) 
    game_json = json.loads(game_call.text)['results']

    for game in game_json:
        game_data[game['name']] = {'id': game['id'], 
                            'name': game['name'],
                            'deck': game['deck'],
                            'description': game['description'], 
                            'genres': game['genres'], 
                            'themes': game['themes'], 
                            'franchises': game['franchises'], 
                            'recommended' : 0} # used in y_true
    return game_data

"""
get_gamespot_games
Use get_gamespot_game_info() as a helper function to return dataset of games
Arguments: 
    api_key (string): API key for GameSpot
    headers (string): specify User-Agent field
    game_count (int): how many pages desired (roughly 100 games per page)
Returns:
    games (dict): {k: game name, v: {game properties, recommend_boolean}}
"""
def get_gamespot_games(api_key, headers, game_count=10):
    # https://stackoverflow.com/questions/38987/how-do-i-merge-two-dictionaries-in-a-single-expression-in-python
    games = {}
    for i in range(game_count):
        new_offset = i*99
        new_games = get_gamespot_game_info(api_key=api_key, headers=headers, offset=new_offset)
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
get_giantbomb_game_info
Get information from GiantBomb API for a given query game
Arguments:
    api_key (string): API key for GameSpot
    query (string): query to search within API call
    headers (string): specify User-Agent field
    session (CachedSession): optional session to store results in local cache
Returns: 
    result (dict): {k: game name, v: {game properties, similar_games, recommend_boolean}}
"""
def get_giantbomb_game_info(api_key, query, headers, session=my_session):

    # https://www.giantbomb.com/api/documentation/#toc-0-17
    search_api_url = "https://www.giantbomb.com/api/search/?api_key=" + api_key + \
        "&format=json&query=" + query + \
        "&resources=game" + \
        "&resource_type=game" 
        # resources = game details that should inform the results, while resource_type = game recommendation itself
    search_game_resp = session.get(search_api_url, headers=headers)

    search_json = json.loads(search_game_resp.text)
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
        'recommended' : 0,
        'similar_games': []
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
    platforms = game_results['platforms']
    GUID = game_results['guid']

    # get aspects of GUID - genres, themes, franchises
    game_api_url = "https://www.giantbomb.com/api/game/" + \
    GUID + "/?api_key=" + api_key + \
        "&format=json"
    game_api_resp = session.get(game_api_url, headers=headers)
    game_api_json = json.loads(game_api_resp.text)
    game_api_results = game_api_json['results']

    # set unexpected input to empty string, and manually set cosine similarity to 0 later to account for it
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

    # develop game dict to return
    query_game_dict[name] = { 
        'name': name,
        'deck': deck,
        'description': desc, 
        'genres': query_genre, 
        'themes': query_theme, 
        'franchises': query_franchise, 
        'recommended': 0
    }

    # find similar games
    # FIXME get csv titles and gamespot titles
    # still have 1 big dataset
    # csv titles will be rec Yes
    # gamespot titles will be rec No
    #
    similar_games_to_query = game_api_json['results']['similar_games']

    sample_similar_games = []
    similar_found = True

    if similar_games_to_query == None:
        sample_similar_games = []
        similar_found = False
    else:
        for i in range(len(similar_games_to_query)):
            name = similar_games_to_query[i]['name']
            guid_val = similar_games_to_query[i]['api_detail_url'][35:-1] # check pdb for confirmation
            sample_similar_games.append({name: guid_val})

        if len(sample_similar_games) == 0 or similar_found == False:
            sample_similar_games = []

    if len(sample_similar_games == 0):
        query_game_dict['similar_games'] = []
    else:
        # FIXME get similar game entries from API call
        pass

                