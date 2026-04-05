from flask import Flask, render_template, request
import pickle

app = Flask(__name__)

model = pickle.load(open("model.pkl", "rb"))

# load features
with open("features.txt") as f:
    features = f.read().splitlines()

@app.route('/')
def home():
    return render_template("index.html", features=features)

@app.route('/predict', methods=['POST'])
def predict():
    values = [float(request.form.get(f)) for f in features]
    prediction = model.predict([values])[0]

    return render_template(
        "index.html",
        features=features,
        prediction_text=f"Estimated Car Price: ${prediction:,.2f}"
    )

if __name__ == "__main__":
    app.run()
