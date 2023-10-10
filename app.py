from flask import Flask, request, jsonify
import pickle
import requests
import pandas as pd

app = Flask(__name__)

@app.route('/<model_name>/predict')
def predict(model_name):
    obs = request.get_json()
    x = pd.get_dummies(pd.DataFrame(obs))
    model = pickle.load(open(f'{model_name}.pkl', 'rb'))
    pred = model.predict(x)
    return jsonify(pred)

if __name__ == 'main':
    app.run(port=5000, debug=True)