
from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

model = pickle.load(open("model.pkl", "rb"))

features = ['ID', 'Price', 'Prod. year', 'Cylinders']

@app.route('/')
def home():
    return render_template("index.html", features=features)

@app.route('/predict', methods=['POST'])
def predict():
    values = [float(request.form.get(f)) for f in features]
    prediction = model.predict([values])
    return render_template("index.html", features=features,
                           prediction_text=f"Predicted Price: {prediction[0]:.2f}")

if __name__ == "__main__":
    app.run()
