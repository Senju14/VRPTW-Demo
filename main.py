from flask import Flask, send_from_directory
from flask_cors import CORS
import os
import sys

sys.path.append(os.path.dirname(__file__))

app = Flask(__name__)
CORS(app)

from src.backend.api import api
app.register_blueprint(api)

@app.route('/')
def index():
    return send_from_directory('src/frontend', 'index.html')

@app.route('/<path:path>')
def serve_frontend(path):
    return send_from_directory('src/frontend', path)

if __name__ == '__main__':
    print("=" * 50)
    print("VRPTW Solver Comparison System")
    print("=" * 50)
    print("Server starting on http://localhost:5000")
    print("=" * 50)
    app.run(debug=True, port=5000, host='0.0.0.0')
    