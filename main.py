from flask import Flask, send_from_directory
from flask_cors import CORS
import os

app = Flask(__name__)
CORS(app)

from src.backend.api import api
app.register_blueprint(api)

@app.route('/')
def index():
    return send_from_directory('src/frontend', 'index.html')

@app.route('/<path:path>')
def serve_static(path):
    return send_from_directory('src/frontend', path)

if __name__ == '__main__':
    app.run(debug=True, port=5000)
    