# app.py
from flask import Flask, render_template, request
import numpy as np
import joblib
import pandas as pd

app = Flask(__name__)

# Load model and feature columns
model = joblib.load('xgb_price_model.pkl')
feature_columns = joblib.load('features.pkl')

# Load raw CSV data to build accurate brand-model mapping
df_raw = pd.read_csv("used_cars.csv")
df_raw = df_raw.dropna(subset=["brand", "model"])
brand_model_map = df_raw.groupby('brand')['model'].unique().apply(list).to_dict()

# Extract encoded choices from feature columns
def extract_choices(columns, prefix):
    return sorted(col.replace(prefix + '_', '') for col in columns if col.startswith(prefix + '_'))

brands = extract_choices(feature_columns, 'brand')
models = extract_choices(feature_columns, 'model')
fuel_types = extract_choices(feature_columns, 'fuel_type')
transmissions = extract_choices(feature_columns, 'transmission')
ext_colors = extract_choices(feature_columns, 'ext_col')
int_colors = extract_choices(feature_columns, 'int_col')

def build_input_vector(form):
    input_dict = dict.fromkeys(feature_columns, 0)
    input_dict['model_year'] = int(form['model_year'])
    input_dict['milage'] = int(form['milage'])
    input_dict['car_age'] = 2025 - int(form['model_year'])
    input_dict['accident'] = int(form['accident'])
    input_dict['clean_title'] = int(form['clean_title'])

    for prefix in ['fuel_type', 'transmission', 'brand', 'model', 'ext_col', 'int_col']:
        key = f"{prefix}_{form[prefix]}"
        if key in input_dict:
            input_dict[key] = 1

    return np.array([list(input_dict.values())])

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    if request.method == 'POST':
        X_input = build_input_vector(request.form)
        prediction = model.predict(X_input)[0]

    return render_template(
        'index.html',
        brands=brands,
        models=models,
        fuel_types=fuel_types,
        transmissions=transmissions,
        ext_colors=ext_colors,
        int_colors=int_colors,
        brand_model_map=brand_model_map,
        prediction=prediction
    )

if __name__ == '__main__':
    app.run(debug=True)