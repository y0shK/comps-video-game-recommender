"""
Create and run test cases to examine SVM output
"""

from sklearn.decomposition import PCA
from get_api_info import get_similar_games, get_giantbomb_game_info
from process_recs import process_text, get_embedding, check_valid_deck_and_desc
import pdb

"""
get_pca_fit
Given a query game, obtain fitted PCA
Arguments:
    query (string): query game title
    model: tfm (encoder model),
    gamespot_key (string): API key for GameSpot
    giantbomb_key (string): API key for GiantBomb
    headers (string): specify User-Agent field
    session (CachedSession): optional session to store results in local cache
Returns:
    pca_embed (word embedding)
"""
def get_pca_fit(query, model, gamespot_key, giantbomb_key, headers, session):

    pca = PCA(n_components = 2) # we want 2 dimensions to visualize

    sim_game = get_similar_games(api_key=gamespot_key, query=query, headers=headers, max_similar=5, session=session)
    sim_game_deck = ''
    sim_game_desc = ''

    for k, v in sim_game.items():

        if check_valid_deck_and_desc(v['deck'], v['description']):
            sim_game_deck += v['deck']
            sim_game_desc += v['description']
        else:
            pass # cannot be recommended with missing information

    if sim_game_deck == '' or sim_game_desc == '':
        return -1

    sim_tokenized_list = process_text(sim_game_deck + sim_game_desc)
    sim_embedding = get_embedding(model, sim_tokenized_list)

    test1_query_info = get_giantbomb_game_info(api_key=giantbomb_key, query=query, headers=headers, session=session)
   
    for k, v in test1_query_info.items():
        test1_query_name = v['name']

    t1qdeck = test1_query_info[test1_query_name]['deck']
    t1qdesc = test1_query_info[test1_query_name]['description']

    query_tokenized_list = process_text(t1qdeck + t1qdesc)
    query_embed = get_embedding(model, query_tokenized_list)

    # fit PCA on query and similar game
    pca.fit([query_embed] + [sim_embedding])
    return pca

def get_transform_embed(rec, model, pca, giantbomb_key, headers, session):
    rec_info = get_giantbomb_game_info(api_key=giantbomb_key, query=rec, headers=headers, session=session)
    for k, v in rec_info.items():
        r_name = v['name']

    deck = rec_info[r_name]['deck']
    desc = rec_info[r_name]['description']

    tokenized_list = process_text(deck + desc)
    rec_embed = get_embedding(model, tokenized_list)

    # transform PCA on potential recommendation game
    rec_lowdim = pca.transform([rec_embed])
    return rec_lowdim

"""
run_testcase
Given a query game and recommendation game contingent on query, return whether game should be recommended
Arguments:
    query (string): query game title
    rec (string): game title of potential recommendation
    model: tfm (encoder model),
    clf: SVM model
    gamespot_key (string): API key for GameSpot
    giantbomb_key (string): API key for GiantBomb
    headers (string): specify User-Agent field
    session (CachedSession): optional session to store results in local cache
Returns:
    rec_pred (int): 1 if recommended, 0 if not
"""
def run_testcase(query, rec, model, clf, gamespot_key, giantbomb_key, headers, session):

    fit_pca = get_pca_fit(query, model, gamespot_key, giantbomb_key, headers, session)

    # check to see if pca is actual pca model, or if invalid input returned 0
    if fit_pca == -1:
        print(query, "not enough info to recommend")
        return 0

    print("PCA check")
    pdb.set_trace()

    rec_info = get_giantbomb_game_info(api_key=giantbomb_key, query=rec, headers=headers, session=session)
    for k, v in rec_info.items():
        r_name = v['name']

    deck = rec_info[r_name]['deck']
    desc = rec_info[r_name]['description']

    tokenized_list = process_text(deck + desc)
    rec_embed = get_embedding(model, tokenized_list)

    # transform PCA on potential recommendation game
    rec_lowdim = fit_pca.transform([rec_embed])
    rec_pred = clf.predict(rec_lowdim)
    print('Given', query, 'recommend', rec, ':', rec_pred)

    print("recommendation pca")
    print(rec_lowdim)

    print("clf prediction based on query")
    return rec_pred