from flask import Flask
import json
from PIL import Image
import numpy as np
import pickle

model = pickle.load("neural_net.p")

app = Flask( __name__ )

@app.route("/")
def hello():
    return "<p>Hello World!</p>"

@app.route("/predict", methods=['POST'])
def predict():
    if request.method == "POST":
        pass
    pass

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port="3210")