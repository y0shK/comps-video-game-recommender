"""
Process aspects of recommendation to feed into algorithm
"""
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
import tensorflow as tf
from sklearn.metrics.pairwise import cosine_similarity

tokenizer = RegexpTokenizer(r'\w+')
stops = set(stopwords.words("english"))

"""
process_text
Take game text (deck/description) and return tokenized version ready for use with cosine similarity
Arguments: game_text (str)
Returns: tokenized_text (str)
"""
def process_text(game_text):
    text_words = list(set(tokenizer.tokenize(game_text)) - stops)
    tokenized_text = [d.lower() for d in text_words if d.isalpha() and len(d) > 1]
    return tokenized_text

"""
check_for_valid_qualities
Take each piece of game information and determine if the given game from the dataset is a valid recommendation.
Invalid recommendations are any that have incomplete information
Arguments: name, deck, desc (str)
        genres, themes, franchises (list of str)
Returns: valid (boolean) determining if game is a possible recommendation
"""
def check_for_valid_qualities(name, deck, desc, genres, themes, franchises):
    if name is None or name == "":
        return False
    
    if deck is None or deck == "":
        return False

    if desc is None or desc == "":
        return False
    
    # length 1 of genres, themes, franchises are OK, but no nulls or empty strings should be processed
    if genres is None or len(genres) < 1 or '' in genres:
        return False
    
    if themes is None or len(themes) < 1 or '' in themes:
        return False
    
    if franchises is None or len(franchises) < 1 or '' in franchises:
        return False
    
    return True

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
