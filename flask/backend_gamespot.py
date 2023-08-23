import requests_cache
import pdb
import json
import os
from dotenv import load_dotenv
import time

from video_game import VideoGame, VideoGameCollection
from tf_idf import get_fitted_train_matrix, get_unfitted_review_matrix
from sklearn.metrics.pairwise import cosine_similarity

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

# wrap recommendation process into helper functions; then, if no recs found, lower cos similarity and try again

"""
perform_vectorization
Arguments: query_string: str (user input that is provided in app.py)
            cos_similarity_threshold: float (what value to assign for cos similarity of user vector/each review vector)
Returns: game_title_ids: {} - key: game title, value: game id
"""
def perform_vectorization(query_string, cos_similarity_threshold):

    # https://stackoverflow.com/questions/38987/how-do-i-merge-two-dictionaries-in-a-single-expression-in-python
    reviews = {}
    for i in range(30): # TODO parameterize this amount
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
    cos_threshold = cos_similarity_threshold

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
    #print(game_title_ids.keys()) e.g., ['Pokemon Blue', 'Pokemon Silver']
    #print(game_title_ids[list(game_title_ids.keys())[0]]) e.g., game_title_ids['Pokemon Blue'] = some_id

    return game_title_ids

"""
extract_recommendations_from_vectors
Arguments: game_title_ids: {} - {game title: game id to cross-reference with review}
Returns: image_recs: {} - {game_id: 'game_id', url: 'url'}
"""
def extract_recommendations_from_vectors(game_title_ids):

    image_recs = []
    for k, v in game_title_ids.items():
        rec_url = "http://www.gamespot.com/api/games/?api_key=" + GAMESPOT_API_KEY + "&format=json" + \
        "&filter=id:" + str(v)

        print("rec_url")
        print(rec_url)

        rec_call = session.get(rec_url, headers=HEADERS)
        rec_json = json.loads(rec_call.text)['results']
        # pdb.set_trace()

        rec = {}

        rec['game'] = rec_json[0]['name']
        rec['url'] = rec_json[0]['image']['original'] 
        image_recs.append(rec)
        # {'Game name': 'https://www.gamespot.com/a/uploads/original/image.png'}

    # pdb.set_trace()
    print(image_recs)
    return image_recs


"""
main()
Arguments: query_string: str (user input raw string)
            cos_similarity_threshold: float 
            (recursively whittle down until 0. Divide by 2 each time. Use round() to ensure base case 0 reached.)
            If 0 reached, truly no results)
Returns: {game: 'gameName', url: 'urlPath'} to Flask API
"""
def main(query_string, cos_similarity_threshold): 

    cos_similarity_threshold = round(cos_similarity_threshold, 2)

    game_title_ids = perform_vectorization(query_string=query_string, cos_similarity_threshold=cos_similarity_threshold)
    image_recs = extract_recommendations_from_vectors(game_title_ids=game_title_ids)
    
    # https://stackoverflow.com/questions/1557571/how-do-i-get-time-of-a-python-programs-execution
    elapsed_time = time.time() - start_time
    print("Time (seconds): ", elapsed_time)

    # recursive call of main()
    if len(image_recs) > 0: # base case 1. we have a recommendation
        print("cos similarity: ", cos_similarity_threshold)
        return image_recs[0]
    # base case 2. we have no possible recommendation. this is like Google saying "no results found"
    if round(cos_similarity_threshold, 2) == 0.0: 
        return {'game name': 'null', 'url': 'null'}
    else: # recursive case. halve cosine similarity and try again.
        return main(query_string=query_string, cos_similarity_threshold=round((cos_similarity_threshold / 2), 2))

"""
Test cases.
(Actual call to main() is in app.py in flask backend)
"""
# main("masterpiece aesthetic gothic", 0.75)
# main("challenging atmospheric", 0.8)
# main("accessible cozy comforting", 0.8)
# main("cozy comforting relaxing", 0.8)
# main("steam deck compatible", 0.8)
# main("final fantasy", 0.8)