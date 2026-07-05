from flask import Flask, render_template, request
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import pickle
import os

app = Flask(__name__)

MODEL_PATH = 'models/model.pkl'
DATA_PATH = 'dataset/dataset.csv'

def convert_sqft_to_num(x):
    # '2100 - 2850' -> 2475 ga marchadam
    tokens = x.split('-')
    if len(tokens) == 2:
        return (float(tokens[0]) + float(tokens[1]))/2
    try:
        return float(x)
    except:
        return None # convert cheyaleni value ante None

# Model train cheyadam - only once
if not os.path.exists(MODEL_PATH):
    df = pd.read_csv(DATA_PATH)

    # Step 1: total_sqft clean cheyadam
    df['total_sqft'] = df['total_sqft'].apply(convert_sqft_to_num)

    # Step 2: bhk number teyadam
    df['bhk'] = df['size'].apply(lambda x: int(x.split(' ')[0]) if isinstance(x, str) else None)

    # Step 3: Empty/null rows motham remove cheyadam
    df = df.dropna(subset=['size', 'total_sqft', 'bath', 'price', 'bhk'])

    # Step 4: Features & Target
    X = df[['total_sqft', 'bhk', 'bath']]
    y = df['price']

    model = LinearRegression()
    model.fit(X, y)

    os.makedirs('models', exist_ok=True)
    with open(MODEL_PATH, 'wb') as f:
        pickle.dump(model, f)
    print("Model trained and saved successfully!")
else:
    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)
    print("Model loaded from file.")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    sqft = float(request.form['sqft'])
    bhk = int(request.form['bhk'])
    bath = int(request.form['bath'])

    prediction = model.predict([[sqft, bhk, bath]])
    return f'Estimated House Price: ₹ {round(prediction[0], 2)}'

if __name__ == '__main__':
    app.run(debug=True)
