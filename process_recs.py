"""
Process aspects of recommendation to feed into algorithm
"""
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
import tensorflow as tf
from sklearn.metrics.pairwise import cosine_similarity

import numpy as np

tokenizer = RegexpTokenizer(r'\w+')
stops = set(stopwords.words("english"))

"""
process_text
Take game text (deck/description) and return tokenized version ready for use with word embeddings
Arguments: game_text (str)
Returns: tokenized_text (list)
"""
def process_text(game_text):
    text_words = list(set(tokenizer.tokenize(game_text)) - stops)
    tokenized_text = [d.lower() for d in text_words if d.isalpha() and len(d) > 1]
    return tokenized_text

"""
check_valid_name
Check to make sure title is valid
Invalid information is any name that is null or empty
Arguments: name (str)
Returns: valid (boolean)
"""
def check_valid_name(name):
    if name == None or name == '':
        return False
    
    return True

"""
check_valid_deck_and_desc
Check to make sure that string inputs to the word embeddings (deck and description) are valid.
Invalid information is any quality that is null or empty
Arguments: deck, desc (str)
Returns: valid (boolean)
"""
def check_valid_deck_and_desc(deck, desc):
    if deck == None or deck == '':
        return False
    if desc == None or desc == '':
        return False
    
    return True

"""
return_valid_review
Return tokenized review to add to word embedding if possible, else return empty list (uninformative)
Arguments: review (str)
Returns: valid_review (list of tokenized str or empty list)
"""
def return_valid_review(review):
    if review != '':
        return process_text(review)
    else:
        return []

"""
check_valid_demographics
Check if each description of any given game demographic (genre, theme) are valid.
Invalid information is any quality that is null or empty as a list
Arguments: li (list)
Returns: valid (boolean)
"""
def check_valid_demographics(li):
    if li == None or len(li) == 0:
        return False
    if li[0] == '':
        return False
    return True

"""
check_for_valid_qualities
Take each piece of game information and determine if that piece of information is usable for recommendation.
Invalid information is any quality that is null or empty
Arguments: name, deck, desc (str)
        genres, themes, franchises (list of str)
Returns: valid_dict (dict) for all keys in arguments. 
E.g., valid_dict['name'] : True if name is not null and not empty
"""
def check_for_valid_qualities(name, deck, desc, genres, themes, franchises):
    valid_dict = {'name': True,
                  'deck': True,
                  'description': True,
                  'genres': True,
                  'themes': True,
                  'franchises': True}

    if name is None or name == "":
        valid_dict['name'] = False
    
    if deck is None or deck == "":
        valid_dict['deck'] = False

    if desc is None or desc == "":
        valid_dict['description'] = False
    
    # length 1 of genres, themes, franchises are OK, but no nulls or empty strings should be processed
    if genres is None or len(genres) < 1 or '' in genres:
        valid_dict['genre'] = False
    
    if themes is None or len(themes) < 1 or '' in themes:
        valid_dict['genre'] = False
    
    if franchises is None or len(franchises) < 1 or '' in franchises:
        valid_dict['genre'] = False
    
    return valid_dict

"""
get_embedding_similarity
Given an instance of TensorFlow Universal Encoder model, 
calculate cosine similarity between two (tokenized and preprocessed) sentence embeddings
Arguments: tfm (encoder model),
            l1 (list of strings): tokenized string to embed (from outer loop)
            l2 (list of strings): tokenized string to embed (from inner loop)
Returns: cos_sim (float): float between 0 and 1
"""
def get_embedding_similarity(tfm, l1, l2):

    str1 = " ".join(l1)
    str2 = " ".join(l2)

    embed1 = tfm([str1])
    embed2 = tfm([str2])
    
    norm1 = tf.nn.l2_normalize(embed1, axis=1)
    norm2 = tf.nn.l2_normalize(embed2, axis=1)

    # cosine_similarity(norm1, norm2) returns [[float]]
    return abs(cosine_similarity(norm1, norm2)[0][0])

"""
get_embedding
Given an instance of TensorFlow Universal Encoder model, 
obtain numpy array value of (tokenized and preprocessed) sentence embedding
Arguments: tfm (encoder model),
            li (list of strings): tokenized string to embed
Returns: np_embed (numpy array): word embedding in np.array form
"""
def get_embedding(tfm, li):
    str1 = " ".join(li)
    embed1 = tfm([str1])
    norm1 = tf.nn.l2_normalize(embed1, axis=1)
    np_embed = np.array(norm1[0])

    return np_embed