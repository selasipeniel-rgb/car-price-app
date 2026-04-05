#!/usr/bin/env bash
pip install -r requirements.txt
pip install jupyter nbconvert
jupyter nbconvert --to notebook --execute car_prediction_final.ipynb --output car_prediction_final.ipynb
