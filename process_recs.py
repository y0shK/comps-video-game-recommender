"""
Process aspects of recommendation to feed into algorithm
"""
from nltk.tokenize import RegexpTokenizer
from nltk import word_tokenize
from nltk.corpus import stopwords
import string
import tensorflow as tf
import numpy as np
tokenizer = RegexpTokenizer(r'\w+')
stops = set(stopwords.words("english"))

"""
process_text
Take game text (deck/description) and return tokenized version ready for use with word embeddings
Arguments: game_text (str)
Returns: tokenized_text (list of strings)
"""
def process_text(game_text):
    text_words = list(set(tokenizer.tokenize(game_text)) - stops)
    tokenized_text = [d.lower() for d in text_words if d.isalpha() and len(d) > 1]
    return tokenized_text

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
get_embedding
Given an instance of TensorFlow Universal Encoder model, 
obtain numpy array value of sentence embedding.
Provide tokenized string to embed
Arguments: tfm (encoder model),
            li (list of strings): tokenized string to embed - defaults to Falsy value
Returns: np_embed (numpy array): word embedding in np.array form
"""
def get_embedding(tfm, li=[]):
    if li: # use tokenized string
        str1 = " ".join(li)
        embed1 = tfm([str1])
    else:
        return []
    
    norm1 = tf.nn.l2_normalize(embed1, axis=1)
    np_embed = np.array(norm1[0])
    return np_embed

"""
preprocess_review
Given a free-text review r, word tokenize it, remove stopwords, remove punctuation.
Return stitched-together string with non-informatic words (articles, punctuation, etc.) removed
Arguments:
    review (str): free-text review for a game g
Returns:
    stitched_review (str): preprocessed review for game g
"""
def preprocess_review(review):
    wt = word_tokenize(review)
    wt = [w.lower() for w in wt if w not in stops and w not in string.punctuation]
    stitched_review = " ".join(wt)
    return stitched_review

"""
get_adjective_context_pairs
Given a preprocessed review pr, get adjective-context pairs (adj + noun) from pr as a list of tokenized strings
Arguments:
    spacy_core (spacy core): define English spacy pipeline for adjective pairs
    pr (str): preprocessed free-text review for a game g
Returns:
    adj_noun_pairs (list): a list of strings containing adj + noun pairs
Example run:
    "Favorite game time dazzling..." -> function -> ['amazing atmosphere', 'enjoyable combat', ...]
"""
def get_adjective_context_pairs(en_nlp, pr, upper_limit=1000000): # can extend en_nlp.max_length if needed)
    doc_pr = en_nlp(pr[0:upper_limit]) # <class 'spacy.tokens.doc.Doc'>

    # get noun chunks from spacy doc, then tie adjectives to noun contexts
    pr_noun_chunks = doc_pr.noun_chunks
    chunk_contexts = []
    adj_noun_pairs = []

    # https://stackoverflow.com/questions/67821137/spacy-how-to-get-all-words-that-describe-a-noun
    for chunk in pr_noun_chunks:
        out_dict = {}
        noun = chunk.root
        if noun.pos_ != 'NOUN':
            continue
        out_dict['noun'] = noun
        for tok in chunk:
            if tok != noun:
                pos_str = str(tok.pos_).lower()
                out_dict[pos_str] = tok
        chunk_contexts.append(out_dict)

    # structure of chunk_contexts:
    # {'noun': 'atmosphere', 'adj': amazing, 'adv': maybe, ...}

    # now tie together words
    for context in chunk_contexts:
        if 'noun' in context.keys() and 'adj' in context.keys():
            phrase = str(context['adj']).lower() + " " + str(context['noun']).lower()
            adj_noun_pairs.append(phrase)
    return adj_noun_pairs
