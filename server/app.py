from flask import Flask
from flask import request, jsonify
import tensorflow as tf

from src.pipeline import predict

graph = tf.get_default_graph()
app = Flask(__name__)


@app.route('/predict', methods=["POST"])
def serve_request():
    global graph
    with graph.as_default():
        vector = request.get_json()
        classified_vector = predict(vector)
        classified_vector['class'] = int(classified_vector['class'])
        return jsonify(classified_vector)


if __name__ == "__main__":
    app.run(host='0.0.0.0')
