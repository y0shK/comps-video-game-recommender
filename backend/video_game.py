"""
video_game.py

Create custom classes to structure the data from the API call (GameSpot)
- VideoGame includes title and review information
- VideoGameCollection stores each VideoGame as a key-value pair in a hash table
- key = game name
- value = list of reviews. 
Each review is a dictionary with game information
    {game_id (int): id
    review (string): review plaintext
    score (float): score given from review}
    
"""

class VideoGame:

    def __init__(self, name, id, review, score):
        self.name = name
        self.id = id
        self.review = review
        self.score = score
        
class VideoGameCollection:

    def __init__(self):
        self.data = {}

    def add_to_collection(self, game: VideoGame):
        if game.name not in self.data.keys():
            self.data[game.name] = {
                'id' : game.id,
                'review': game.review,
                'score': game.score
            }
        
    def to_string(self):
        for k, v in self.data.items():
            print("Title: ", k)
            print("Review: ", v['review'])
            print("Score: ", v['score'])

    def get_data(self):
        return self.data