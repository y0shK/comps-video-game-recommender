from gensim.test.utils import common_texts
from gensim.corpora.dictionary import Dictionary
from gensim.models import LdaModel
import pdb

"""
Inputs: list of strings that involve training data (want pretrained LDA model)
Outputs: a list of topics that cohesively combine all documents, and a probability distribution
that indicates how well each document fits the 
"""

"""
Train the LDA model on deck and description documents of the query game
Then get the topics from the deck/description elements
Then go through dataset games
    Check how well each dataset document fits each topic suggested by LDA
    Check if any topic is in deck/description from review
    If probability distribution > some threshold, add the game
"""

# https://radimrehurek.com/gensim/models/ldamodel.html
# https://radimrehurek.com/gensim/corpora/dictionary.html - doc2bow
# https://en.wikipedia.org/wiki/Latent_Dirichlet_allocation
# https://stackoverflow.com/questions/952914/how-do-i-make-a-flat-list-out-of-a-list-of-lists

# common_texts is a list of lists of documents
common_dictionary = Dictionary(common_texts)

# set common_texts into Gensim dictinary
# use bag of words on each list of documents in common_texts
common_corpus = [common_dictionary.doc2bow(text) for text in common_texts]

pdb.set_trace()

# Train the model on the corpus.

# find topics amongst bag of words of documents
lda = LdaModel(common_corpus, num_topics=10)

pdb.set_trace()

# get a new list of lists of documents
# and check how well each topic fits each document
# this is the probability distribution
other_texts = [

    ['computer', 'time', 'graph'],
    ['survey', 'response', 'eps'],
    ['human', 'system', 'computer']

]

pdb.set_trace()

# bag of words the new list of list of topics
# doc2bow: (token_id, token_count)
# go through each list in other_texts
# for each list, get a tuple of token_id and token_count
other_corpus = [common_dictionary.doc2bow(text) for text in other_texts]

# unseen_doc takes tuple lists from other_corpus
# then, LDA is performed on that tuple list
unseen_doc = other_corpus[0] # unseen_doc is the first entry in other_corpus; the BoW format of computer time graph

vector = lda[unseen_doc]  # for a given document from unseen_doc, check to see how much each topic relates to the whole doc

pdb.set_trace()