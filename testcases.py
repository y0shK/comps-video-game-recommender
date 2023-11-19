"""
Create and run test cases to examine SVM output
"""

from get_api_info import get_similar_games, get_giantbomb_game_info
from process_recs import process_text, get_embedding, check_valid_deck_and_desc

"""
run_testcase
Given a query game and recommendation game contingent on query, return whether game should be recommended
Arguments:
    query (string): query game title
    rec (string): game title of potential recommendation
    model: tfm (encoder model),
    pca: principal component analysis model
    clf: SVM model
    gamespot_key (string): API key for GameSpot
    giantbomb_key (string): API key for GiantBomb
    headers (string): specify User-Agent field
    session (CachedSession): optional session to store results in local cache
Returns:
    rec_pred (int): 1 if recommended, 0 if not
"""
def run_testcase(query, rec, model, pca, clf, gamespot_key, giantbomb_key, headers, session):

    sim_game = get_similar_games(api_key=gamespot_key, query=query, headers=headers, max_similar=1, session=session)
    sim_game_deck = ''
    sim_game_desc = ''

    for k, v in sim_game.items():

        if check_valid_deck_and_desc(v['deck'], v['desc']):
            sim_game_deck = v['deck']
            sim_game_desc = v['description']
        else:
            return 0 # cannot be recommended with missing information

    sim_tokenized_list = process_text(sim_game_deck + sim_game_desc)
    sim_embedding = get_embedding(model, sim_tokenized_list)

    test1_query_info = get_giantbomb_game_info(api_key=giantbomb_key, query=query, headers=headers, session=session)
    test1_rec_info = get_giantbomb_game_info(api_key=giantbomb_key, query=rec, headers=headers, session=session)

    for k, v in test1_query_info.items():
        test1_query_name = v['name']

    for k, v in test1_rec_info.items():
        test1_rec_name = v['name']

    t1qdeck = test1_query_info[test1_query_name]['deck']
    t1qdesc = test1_query_info[test1_query_name]['description']

    t1rdeck = test1_rec_info[test1_rec_name]['deck']
    t1rdesc = test1_rec_info[test1_rec_name]['description']

    query_tokenized_list = process_text(t1qdeck + t1qdesc)
    rec_tokenized_list = process_text(t1rdeck + t1rdesc)

    query_embed = get_embedding(model, query_tokenized_list)
    rec_embed = get_embedding(model, rec_tokenized_list) 

    # fit PCA on query and similar game
    pca.fit([query_embed] + [sim_embedding])
    # transform PCA on potential recommendation game
    rec_lowdim = pca.transform([rec_embed])
    rec_pred = clf.predict(rec_lowdim)
    print('Given', query, 'recommend', rec, ':', rec_pred)

    return rec_pred