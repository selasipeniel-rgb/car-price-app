import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

print("Loading data...")
car_df = pd.read_csv("car_price_prediction.csv")

print("Renaming columns...")
car_df = car_df.rename(columns={
    'Price': 'price',
    'Levy': 'levy',
    'Model': 'model',
    'Prod. year': 'production_year',
    'Fuel type': 'fuel_type',
    'Engine volume': 'engine_volume',
    'Mileage': 'mileage',
    'Cylinders': 'cylinders',
    'Airbags': 'airbags',
    'Manufacturer': 'manufacturer'
})

print("Cleaning data...")
car_df['price'] = pd.to_numeric(car_df['price'], errors='coerce')
car_df['levy'] = car_df['levy'].str.replace('-', '0', regex=False)
car_df['mileage'] = (
    car_df['mileage']
    .str.replace('km', '', regex=False)
    .str.replace(' ', '', regex=False)
    .str.replace(',', '', regex=False)
    .str.strip()
)
car_df['mileage'] = pd.to_numeric(car_df['mileage'], errors='coerce')
car_df['engine_volume'] = (
    car_df['engine_volume']
    .str.replace('Turbo', '', regex=False)
    .str.strip()
)
car_df = car_df.astype({'levy': float, 'engine_volume': float})
car_df.dropna(subset=['price', 'levy', 'mileage', 'engine_volume'], inplace=True)

print("Removing outliers...")
low  = car_df['price'].quantile(0.01)
high = car_df['price'].quantile(0.99)
car_df = car_df[(car_df['price'] >= low) & (car_df['price'] <= high)]
print(f"Rows after cleaning: {len(car_df)}")

print("Encoding categorical features...")
car_df_encoded = pd.get_dummies(car_df, columns=['fuel_type', 'manufacturer'], drop_first=True)

numeric_features = ['production_year', 'levy', 'mileage', 'cylinders', 'airbags', 'engine_volume']
categorical_features = [c for c in car_df_encoded.columns
                        if c.startswith('fuel_type_') or c.startswith('manufacturer_')]
feature_cols = numeric_features + categorical_features

X = car_df_encoded[feature_cols]
y = car_df_encoded['price']

print("Training model...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train[numeric_features] = scaler.fit_transform(X_train[numeric_features])
X_test[numeric_features]  = scaler.transform(X_test[numeric_features])

model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)

from sklearn.metrics import r2_score
y_pred = model.predict(X_test)
print(f"R² Score: {r2_score(y_test, y_pred):.3f}")

print("Saving model files...")
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

with open("features.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(feature_cols))

with open("numeric_features.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(numeric_features))

print("Done! Saved: model.pkl, scaler.pkl, features.txt, numeric_features.txt")
