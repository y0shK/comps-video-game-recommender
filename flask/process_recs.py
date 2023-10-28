"""
Process aspects of recommendation to feed into algorithm
"""
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
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
    
    if genres is None or len(genres) < 1 or '' in genres:
        return False
    
    if themes is None or len(themes) < 1 or '' in themes:
        return False
    
    if franchises is None or len(franchises) < 1 or '' in franchises:
        return False
    
    return True