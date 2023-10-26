"""
Supervised learning project to reverse engineer recommended games from GiantBomb API
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
import pandas as pd
import matplotlib.pyplot as plt

from get_api_info import get_giantbomb_game_info

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
1. form a dataset by combining games from metacritic csv and GameSpot API
Get titles from each of the data sources, then get information from GiantBomb API call
"""
csv_titles = list(set([i for i in csv_titles_df['Title']][0:10]))
print(csv_titles[0])

query_dict = get_giantbomb_game_info(api_key=GIANTBOMB_API_KEY, query=csv_titles[0], headers=HEADERS,session=session)
pdb.set_trace()

# print(query_dict)

# TODO scale up calling get_giantbomb_game_info
# First do it on multiple queries from csv
# Then do it on multiple queries from gamespot
# Then combine in for d1 in dataset, for d2 in dataset as in pseudocode and check performance (ROC)