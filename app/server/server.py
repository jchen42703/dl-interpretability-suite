import os
from flask import Flask
from flask_cors import CORS, cross_origin

app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'

@app.route("/api/pred")
@cross_origin()
def index():
    pred = 41
    return {'pred': pred}
