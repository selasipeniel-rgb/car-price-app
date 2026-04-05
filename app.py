from flask import Flask, render_template, request
import pickle
import pandas as pd

app = Flask(__name__)

# load model, scaler, and feature lists
model = pickle.load(open("model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

with open("features.txt", encoding="utf-8") as f:
    all_features = f.read().splitlines()

with open("numeric_features.txt", encoding="utf-8") as f:
    numeric_features = f.read().splitlines()

# the 6 inputs the user fills in on the form
form_fields = ['production_year', 'levy', 'mileage', 'cylinders', 'airbags', 'engine_volume']

@app.route('/')
def home():
    return render_template("index.html", features=form_fields)

@app.route('/predict', methods=['POST'])
def predict():
    # build a zero-filled row for all model features (handles encoded dummies)
    row = dict.fromkeys(all_features, 0)

    # fill in numeric inputs from the form
    for f in form_fields:
        row[f] = float(request.form.get(f, 0))

    # fill in categorical dummies if user selected them
    fuel_type    = request.form.get('fuel_type', '')
    manufacturer = request.form.get('manufacturer', '')
    if fuel_type and f"fuel_type_{fuel_type}" in row:
        row[f"fuel_type_{fuel_type}"] = 1
    if manufacturer and f"manufacturer_{manufacturer}" in row:
        row[f"manufacturer_{manufacturer}"] = 1

    # build DataFrame to preserve column order
    input_df = pd.DataFrame([row])[all_features]

    # scale numeric features only
    input_df[numeric_features] = scaler.transform(input_df[numeric_features])

    prediction = model.predict(input_df)[0]

    return render_template(
        "index.html",
        features=form_fields,
        prediction_text=f"Estimated Car Price: ${prediction:,.2f}"
    )

if __name__ == "__main__":
    app.run(debug=True)
