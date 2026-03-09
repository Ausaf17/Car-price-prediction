from flask import Flask, render_template, request, redirect
from flask_cors import CORS, cross_origin
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

app = Flask(
    __name__,
    template_folder=str(BASE_DIR / 'template'),
    static_folder=str(BASE_DIR / 'static'),
)
cors = CORS(app)

# Load data and train model once at startup
car = pd.read_csv(BASE_DIR / 'cleaned_car.csv')

X = pd.get_dummies(
    car[['name', 'company', 'year', 'kms_driven', 'fuel_type']],
    drop_first=True
)
y = car['Price']

model = LinearRegression()
model.fit(X, y)
feature_columns = X.columns

@app.route('/', methods=['GET', 'POST'])
def index():
    companies = sorted(car['company'].unique())
    car_models = sorted(car['name'].unique())
    year = sorted(car['year'].unique(), reverse=True)
    fuel_type = car['fuel_type'].unique()

    companies.insert(0, 'Select Company')
    return render_template('index.html', companies=companies, car_models=car_models, years=year, fuel_types=fuel_type)


@app.route('/predict', methods=['POST'])
@cross_origin()
def predict():

    company = request.form.get('company')

    car_model = request.form.get('car_models')
    year = int(request.form.get('year'))
    fuel_type = request.form.get('fuel_type')
    driven = int(request.form.get('kilo_driven'))

    # Build a single-row DataFrame and apply the same encoding as training
    input_df = pd.DataFrame(
        [[car_model, company, year, driven, fuel_type]],
        columns=['name', 'company', 'year', 'kms_driven', 'fuel_type']
    )
    input_encoded = pd.get_dummies(input_df, drop_first=True)
    input_encoded = input_encoded.reindex(columns=feature_columns, fill_value=0)

    prediction = model.predict(input_encoded)
    print(prediction)

    return str(np.round(prediction[0], 2))



if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000, debug=True, use_reloader=False)