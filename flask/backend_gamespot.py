import requests_cache
import pdb
import json
import os
from dotenv import load_dotenv
import time

from video_game import VideoGame, VideoGameCollection

"""
backend_gamespot.py
- Setup API call credentials
- Get response from review GameSpot API request
- Obtain frontend-relevant info (game/image) to pass through Flask API
"""

# http://www.gamespot.com/api/games/?api_key=<YOUR_API_KEY_HERE>

start_time = time.time()

load_dotenv()

session = requests_cache.CachedSession("game_cache")
GAMESPOT_API_KEY = os.getenv('GAMESPOT_API_KEY')
HEADERS = {'User-Agent': 'Video Game Recommender Comprehensive Project'}

def get_game_info(api_key, headers):
    """
    Arguments: api key, headers
    Returns: game_data (dict): key=name and value=
        id, description, genres, themes, franchises, release date, image
    """
    game_url = "http://www.gamespot.com/api/games/?api_key=" + api_key + "&format=json"
    game_call = session.get(game_url, headers=headers)
    
    game_json = json.loads(game_call.text)['results']
    # pdb.set_trace()

    game_data = {}
    for game in game_json:
        game_data[game['name']] = {'id': game['id'], 
                             'description': game['description'], 
                             'genres': game['genres'], 
                             'themes': game['themes'], 
                             'franchises': game['franchises'], 
                             'release_date': game['release_date'],
                             'image': game['image']}
        
    return game_data

def get_review_info(api_key, headers, offset):

    review_url = "http://www.gamespot.com/api/reviews/?api_key=" + api_key + \
    "&format=json" + \
    "&offset=" + str(offset)
    review_call = session.get(review_url, headers=headers)

    review_json = json.loads(review_call.text)['results']
    # pdb.set_trace()

    review_data = {}
    for review in review_json:
        review_data[review['game']['name']] = {'title': review['title'],
                                       'body': review['body'],
                                       'score': review['score'],
                                       'game_id': review['game']['id'],
                                       'game_name': review['game']['name']}

    # pdb.set_trace()

    return review_data

# games = get_game_info(api_key=GAMESPOT_API_KEY, headers=HEADERS)

def main(query_string): 

    # https://stackoverflow.com/questions/38987/how-do-i-merge-two-dictionaries-in-a-single-expression-in-python
    reviews = {}
    for i in range(30):
        new_reviews = get_review_info(api_key=GAMESPOT_API_KEY, headers=HEADERS, offset=i*99)
        reviews = {**reviews, **new_reviews}

    # pdb.set_trace()

    # create new video game collection to store video games used for reviews (TF-IDF)
    game_collection = VideoGameCollection()

    for k, v in reviews.items():
        game = VideoGame(name=v['game_name'], id=v['game_id'], review=v['body'], score=v['score'])
        game_collection.add_to_collection(game=game)

    print(game_collection.get_data())

    # pdb.set_trace()

    from tf_idf import get_fitted_train_matrix, get_unfitted_review_matrix

    from sklearn.metrics.pairwise import cosine_similarity

    gamesData = game_collection.get_data()

    # find cosine similarity between each game and the query
    # highest cosine similarities are recorded and associated games are returned

    # get user query
    # print("Enter a query:")
    # query_string = input()

    # structure query into appropriate data format for tf-idf methods
    user_query = [query_string] 
    user_matrix = get_fitted_train_matrix(user_query) 

    # go through videoGameCollection's data {key: value} pairs
    # find how similar the query vector and review vector are by using cosine similarity
    # if the cosine similarity is greater than a certain threshold, then recommend the game
    # record game titles of recommended games
    game_title_ids = {} # key: game title, value: game id
    cos_threshold = 0.75

    print(len(reviews))
    print(len(gamesData))

    # pdb.set_trace()

    for k, v in gamesData.items():
        
        # include game title in game review
        # this is included because a search query of a specific game should recommend that game
        # similar to what Google does - recommends search query & other related games
        review_content = v['review'] + " " + k
        review_content = [review_content]
        m = get_unfitted_review_matrix(user_query, review_content)

        if cosine_similarity(user_matrix, m) >= cos_threshold:
            game_title_ids[k] = v['id']

    print("Recommended games: ")
    print(game_title_ids.keys())

    # TODO add error checks in case no games recommended
    # get images from names of recommended games - DONE

    image_recs = {}
    for k, v in game_title_ids.items():
        rec_url = "http://www.gamespot.com/api/games/?api_key=" + GAMESPOT_API_KEY + "&format=json" + \
        "&filter=id:" + str(v)
        rec_call = session.get(rec_url, headers=HEADERS)
        rec_json = json.loads(rec_call.text)['results']
        # pdb.set_trace()

        image_recs[rec_json[0]['name']] = rec_json[0]['image']['original'] 
        # {'Game name': 'https://www.gamespot.com/a/uploads/original/image.png'}

    # pdb.set_trace()
    print(image_recs)

    # https://stackoverflow.com/questions/1557571/how-do-i-get-time-of-a-python-programs-execution
    elapsed_time = time.time() - start_time
    print("Time (seconds): ", elapsed_time)

    return image_recs

# main("cute artistic")