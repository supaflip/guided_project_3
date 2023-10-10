from flask import Flask, request, jsonify
import pickle
import requests
import pandas as pd
import numpy as np

# run: flask run

app = Flask(__name__)

@app.route('/<model_name>/predict', methods=['GET', 'POST'])
def predict(model_name):
    obs = request.args.to_dict()
    #print(obs)
    input_features = pd.get_dummies(pd.DataFrame([obs]))
    model = pickle.load(open(f'{model_name}.pkl', 'rb'))
    features, x = np.array([(f, int(f in input_features)) for f in model.feature_names_in_]).transpose()
    print(x)
    pred = model.predict(x.reshape(1, -1))[0]
    print(pred)
    return str(pred)

if __name__ == 'main':
    app.run(port=5000, debug=True)