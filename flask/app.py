from flask import Flask
from flask import render_template
from flask import request
app = Flask(__name__)

from backend_gamespot import main

@app.route("/")
def home():
    return render_template('index.html')

# https://stackoverflow.com/questions/11556958/sending-data-from-html-form-to-a-python-script-in-flask?rq=3
@app.route("/handle_data", methods=["POST"])
def handle_data():
    query = request.form['query']
    image_recs = main(query_string=query)
    return image_recs

    # https://stackoverflow.com/questions/46785507/python-flask-display-image-on-a-html-page

    
