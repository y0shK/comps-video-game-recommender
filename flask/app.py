from flask import Flask
from flask import render_template
from flask import request, jsonify
from flask_cors import CORS, cross_origin
from backend_gamespot import main

app = Flask(__name__)
CORS(app)

# https://stackoverflow.com/questions/72022246/how-to-pass-a-flask-post-to-react

class Query:
    query_body = ''
    def update_query(cls, new_query):
        cls.query_body = new_query

data = Query()

@app.route('/')
@cross_origin("http://localhost:3000")
def home():
    return render_template('hello_world.html')

@app.route('/query', methods=['GET', 'POST'])
@cross_origin("http://localhost:3000")
def query():
    if request.method == "POST":
        query = request.json['query_body']
        data.update_query(query)
        return data.query_body
    elif request.method == "GET":
        return main(data.query_body)