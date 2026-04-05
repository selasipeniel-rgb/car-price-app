from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# load model, scaler, and feature names
model = pickle.load(open("model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

with open("features.txt") as f:
    features = f.read().splitlines()

@app.route('/')
def home():
    return render_template("index.html", features=features)

@app.route('/predict', methods=['POST'])
def predict():
    values = np.array([[float(request.form.get(f)) for f in features]])
    values_scaled = scaler.transform(values)
    prediction = model.predict(values_scaled)[0]
    return render_template(
        "index.html",
        features=features,
        prediction_text=f"Estimated Car Price: ${prediction:,.2f}"
    )

if __name__ == "__main__":
    app.run(debug=True)
