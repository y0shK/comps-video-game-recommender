import requests_cache
import pdb
import json
import os
from dotenv import load_dotenv
import time
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

from vaderSentiment import vaderSentiment
from sklearn.decomposition import TruncatedSVD

from video_game import VideoGame, VideoGameCollection
from tf_idf import get_fitted_train_matrix, get_unfitted_review_matrix
from sklearn.metrics.pairwise import cosine_similarity

"""
backend_gamespot.py
- Setup API call credentials
- Get response from review GameSpot API request
- Perform NLP backend algorithm 
(statistical recommendation with TF-IDF, semantic recommendation with sentiment analysis & LSA)
- Obtain frontend-relevant info (game/image) to pass through Flask API
(main() is called in flask/app.py)
"""
start_time = time.time()
load_dotenv()
session = requests_cache.CachedSession("game_cache")

# http://www.gamespot.com/api/games/?api_key=<YOUR_API_KEY_HERE>
GAMESPOT_API_KEY = os.getenv('GAMESPOT_API_KEY')
HEADERS = {'User-Agent': 'Video Game Recommender Comprehensive Project'}

"""
get_game_info
Provide dictionary-based information about individual games and their qualities (description, themes, etc.)
Arguments: api key, headers
Returns: game_data (dict): key=name and value=
    id, description, genres, themes, franchises, release date, image
"""
def get_game_info(api_key, headers):
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

"""
get_review_info
Provide dictionary-based information about reviews (1 per game)
Arguments: api_key (string): API key for GameSpot
            headers (string): specify User-Agent field
            offset (int): how many pages API should skip before sending new content in response
Returns: review_data (dict): key=game name, value=title, review, score, and game_id
"""

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

"""
clean_user_query
Clean the user query by removing stopwords, converting query to all lowercase, removing punctuation, etc.
TODO also clean user query of offensive content (since input is free-text)
Arguments: query_string: str (user input that is provided in app.py)
Returns: cleaned_query_string: str (remove stopwords with nltk)
"""

# TODO ensure no offensive content is entered (check how to detect offensive content in literature)
def clean_user_query(query_string):
    # nltk tokenize to get rid of query stopwords
    # https://www.nltk.org/api/nltk.tokenize.html
    # https://stackoverflow.com/questions/22763224/nltk-stopword-list
    broken_query = word_tokenize(query_string)
    stops = set(stopwords.words('english'))

    clean_query = ""
    for word in broken_query:
        if word not in stops and word.isalnum(): # check non-stopword and alphanumeric
            clean_query += word.lower() + " "
    clean_query = clean_query.strip() # https://docs.python.org/3/library/stdtypes.html
    
    #print("difference in cleanup")
    #print(clean_query)
    #print(query_string)
    if clean_query:
        return clean_query
    else:
        return query_string

"""
get_reviews
Get review dataset from GameSpot API
Arguments: review_count (how many reviews to query from API)
Returns: {'game1': 'review1', ...}
"""
def get_reviews(review_count=30):
    # https://stackoverflow.com/questions/38987/how-do-i-merge-two-dictionaries-in-a-single-expression-in-python
    reviews = {}
    for i in range(review_count):
        new_reviews = get_review_info(api_key=GAMESPOT_API_KEY, headers=HEADERS, offset=i*99)
        reviews = {**reviews, **new_reviews}
    return reviews

"""
get_query_sentiment
Get the sentiment of the user-provided string query to provide a semantic approach that complements statistical TF-IDF
Arguments: query_string: str (user input provided in app.py)
Returns: query_sentiment: float (sentiment as calculated by VADER)
"""
def get_query_sentiment(query_string):
    analyzer = vaderSentiment.SentimentIntensityAnalyzer()
    vs = analyzer.polarity_scores(query_string)
    print("Polarity: ", str(vs))
    print("Compound polarity: ", str(vs['compound']))
    print(type(vs['compound']))
    return vs['compound']

"""
perform_vectorization
Perform vectorization of both statistical (TF-IDF) and semantic (sentiment analysis/LSA) approaches
Arguments: query_string: str (user input that is provided in app.py)
            cos_similarity_threshold: float (what value to assign for cos similarity of user vector/each review vector)
Returns: game_title_ids: {} - key: game title, value: {game id, cos similarity between user vector/review}
"""
def perform_vectorization(query_string, cos_similarity_threshold):
    # create new video game collection to store video games used for reviews (TF-IDF)
    game_collection = VideoGameCollection()

    # create list of all reviews to perform SVD on TF-IDF matrix
    review_corpus = []

    reviews = get_reviews(review_count=30)
    for k, v in reviews.items():
        game = VideoGame(name=v['game_name'], id=v['game_id'], review=v['body'], score=v['score'])
        game_collection.add_to_collection(game=game)
        review_corpus.append(v['body'])

    print(game_collection.get_data())
    # pdb.set_trace()

    gamesData = game_collection.get_data()

    # structure query into appropriate data format for tf-idf methods
    user_query = [query_string] 
    vectorizer, user_matrix = get_fitted_train_matrix(user_query) 

    # go through videoGameCollection's data {key: value} pairs
    # find how similar the query vector and review vector are by using cosine similarity
    # if the cosine similarity is greater than a certain threshold, then recommend the game
    game_title_ids = {} # key: game title, value: {game id, cosine similarity, review sentiment}
    cos_threshold = cos_similarity_threshold

    print(len(reviews))
    print(len(gamesData))
    # pdb.set_trace()

    # sentiment analysis
    # https://github.com/cjhutto/vaderSentiment
    query_sentiment = get_query_sentiment(query_string=query_string)
    analyzer = vaderSentiment.SentimentIntensityAnalyzer()

    # set up latent semantic analysis
    # https://machinelearninggeek.com/latent-semantic-indexing-using-scikit-learn/
    review_vectorizer, review_train_data = get_fitted_train_matrix(review_corpus)

    # define number of topics per review, and then factorize tf-idf matrix 
    num_components = 10
    lsa = TruncatedSVD(n_components=num_components, n_iter=100, random_state=42)
    lsa.fit_transform(review_train_data)
    # TODO figure out evaluation for SVD
    # sigma = lsa.singular_values_ 
    # v_transpose = lsa.components_.T

    review_topics = review_vectorizer.get_feature_names_out()
    print("review topics")
    print(review_topics)

    review_terms_list = []
    for index, component in enumerate(lsa.components_):
        zipped = zip(review_topics, component)
        top_terms_key = sorted(zipped, key = lambda t: t[1], reverse=True)[:5]
        top_terms_list = list(dict(top_terms_key).keys())
        review_terms_list += top_terms_list
        print("Topic " + str(index) + ": ", top_terms_list)

    for k, v in gamesData.items():
        review_content = v['review'] + " " + k # include game title in case string query is same franchise/IP
        review_content = [review_content]
        unfitted_vectorizer, m = get_unfitted_review_matrix(user_query, review_content)

        # check sentiment of review to see if "emotion" of game is similar
        vs = analyzer.polarity_scores(review_content)
        review_sentiment = vs['compound']
        sentiment_threshold = 0.1

        # https://stackoverflow.com/questions/61956463/extract-data-from-numpy-array-of-shape
        # https://stackoverflow.com/questions/20457038/how-to-round-to-2-decimals-with-python
        cos_sim = cosine_similarity(user_matrix, m) # gives array[[cos_sim]], extract cos_sim

        if cos_sim >= cos_threshold:
            cos_sim_value = round(float(cos_sim.item()), 2)
            game_title_ids[k] = {'game_id': v['id'], 'game_input_cos_similarity': cos_sim_value,
                                 'sentiment': review_sentiment}

        if round(abs(query_sentiment - review_sentiment), 2) <= sentiment_threshold:
            game_title_ids[k] = {'game_id': v['id'], 
                                 'game_input_cos_similarity': round(float(cos_sim.item()), 2),
                                 'sentiment': review_sentiment} 

        # use latent semantic analysis results
        # if query is relevant to overall review LSA and this specific game review, recommend this game
        # bag of words assumption
        query_split = query_string.split()
        for term in query_split:
            if term in review_terms_list and game not in game_title_ids and \
                (term in k or term in v['review']): # term in name (key) or review content
                game_title_ids[k] = {'game_id': v['id'], 
                                 'game_input_cos_similarity': round(float(cos_sim.item()), 2),
                                 'sentiment': review_sentiment} 

    print("Recommended games: ")
    # e.g., game_title_ids['Pokemon Blue'] = {some_id, some_cos_sim}
    return game_title_ids

"""
extract_recommendations_from_vectors
Arguments: game_title_ids: {} - {game title: {game id: game id to cross-reference with review,
                                                cos similarity: float similarity between user vector/review}
Returns: image_recs: list of dicts [{}, {}, ...]
                         each dict is {game_id: 'game_id', url: 'url', cos_sim: x such that 0.0 <= x <= 1.0, sentiment: review sentiment with 0.0 <= x <= 1.0}
"""
def extract_recommendations_from_vectors(game_title_ids):
    # in case no appropriate game recommendations are found, return empty list. 
    # In main(), recursive case will recall process with lower cosine similarity threshold
    if len(game_title_ids) == 0:
        return []

    image_recs = []
    for k, v in game_title_ids.items():
        rec_url = "http://www.gamespot.com/api/games/?api_key=" + GAMESPOT_API_KEY + "&format=json" + \
        "&filter=id:" + str(v['game_id'])
        print("rec_url")
        # print(rec_url)
        rec_call = session.get(rec_url, headers=HEADERS)
        rec_json = json.loads(rec_call.text)['results']
        # pdb.set_trace()

        # only add recommendation to potential list if response object exists
        # this check ensures all information can be found
        if rec_json and rec_json[0]:
            rec = {}
            rec['game'] = rec_json[0]['name']
            rec['url'] = rec_json[0]['image']['original']
            rec['cos_sim'] = v['game_input_cos_similarity'] 
            rec['sentiment'] = v['sentiment']
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
    rec_string = clean_user_query(query_string)
    game_title_ids = perform_vectorization(query_string=rec_string, cos_similarity_threshold=cos_similarity_threshold)
    image_recs = extract_recommendations_from_vectors(game_title_ids=game_title_ids)
    
    # https://stackoverflow.com/questions/1557571/how-do-i-get-time-of-a-python-programs-execution
    elapsed_time = time.time() - start_time
    print("Time (seconds): ", elapsed_time)

    # https://stackoverflow.com/questions/5320871/how-to-find-the-min-max-value-of-a-common-key-in-a-list-of-dicts
    # recursive call of main()
    # base case - we have a recommendation. return the best one, depending on statistical/semantic methods
    if len(image_recs) > 0: 
        print("current cos similarity threshold: ", cos_similarity_threshold)
        max_cos_sim_rec = max(image_recs, key = lambda x : x['cos_sim'])
        max_sentiment_rec = max(image_recs, key = lambda x : x['sentiment'])
        print(max_cos_sim_rec)

        # if every recommendation has cosine similarity 0.0, then use sentiment to break tie
        if max_cos_sim_rec['cos_sim'] == 0.0:
            return max_sentiment_rec
        
        # if there are multiple recommendations with cosine similarity 1.0, use sentiment to break tie
        # https://stackoverflow.com/questions/4587915/return-list-of-items-in-list-greater-than-some-value
        cos_sim_1 = [rec for rec in image_recs if rec['cos_sim'] == 1.0]
        if len(cos_sim_1) == 1:
            return cos_sim_1[0]
        elif len(cos_sim_1) > 1:
            return max_sentiment_rec

        return max_cos_sim_rec

    else: # recursive case. halve cosine similarity and try again.
        return main(query_string=query_string, cos_similarity_threshold=round((cos_similarity_threshold / 2), 2))

"""
Test cases.
(Actual call to main() is in app.py in flask backend)
"""
# main("cute artistic", 0.75)
# main("masterpiece aesthetic gothic", 0.75)
# main("challenging atmospheric", 0.8)
# main("accessible cozy comforting", 0.8)
# main("cozy comforting relaxing", 0.8)
# main("steam deck compatible", 0.8)
# main("final fantasy with customization and atmosphere", 0.75)
# main("chrono trigger", 0.75)
# main("super metroid", 0.75)
# main("aria of sorrow", 0.75) # test stopword removal
# main("WARIOWARE TOUCHED!", 0.75) # test lowercase, punctuation removal
# main("vampire the masquerade dnd like game", 0.8)
# main("customizable system-driven dungeon crawler", 0.75)