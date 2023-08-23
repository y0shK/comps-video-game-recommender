# Video Game Recommender (Senior Comprehensive Project)

## Description
This project is a video game recommender that uses keyword input from the user to deliver video game recommendations. It is built with a React frontend and Flask backend. Part of the Flask backend uses the TF-IDF algorithm. This vectorizes both the string input provided by the user and the title/review dataset content fetched from the GameSpot API. From these vectors, cosine similarity is used to obtain vectors which are most similar to the user-provided vector, and reverse lookup finds game titles to provide. Other aspects of game recommendation are also considered.

## How to Run

## Threats to Validity

## Citations

### API used
* https://www.gamespot.com/api/documentation

## TODO 
[x] add a default route in Flask <br>
[x] clean up user-facing React UI <br>
[x] automatically fire API rather than have to fire localhost:5000/query manually <br>
[x] add recursive cosine similarity polling (each time, rerun with lesser similarity index) to guarantee recommendation <br>
[x] quality check - clean up user input (remove stopwords, remove punctuation, ensure lowercase, etc.) <br> 
[ ] ethical check - clean up user input (ensure no offensive content entered)
[ ] improve recommendation algorithm and transfer from Python to Flask (more recommendations, better error checking if 0 recommendations found, etc.) <br>
[ ] speed up process (currently takes ~30s) <br>
