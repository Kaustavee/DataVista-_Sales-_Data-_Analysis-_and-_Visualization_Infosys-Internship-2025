import matplotlib
matplotlib.use('Agg')
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score
from sklearn.ensemble import RandomForestRegressor
from flask import Flask, render_template, request, flash
from flask_caching import Cache
from io import BytesIO
import base64

import matplotlib.pyplot as plt

app = Flask(__name__)
app.secret_key = 'your_secret_key'  # Needed for flashing messages
cache = Cache(app, config={'CACHE_TYPE': 'SimpleCache'})

@cache.memoize()
def load_data():
    try:
        return pd.read_csv('Energy_consumption_by_fuel_EJ.csv')
    except FileNotFoundError:
        print("Error: Energy_consumption_by_fuel_EJ.csv not found.")
        return pd.DataFrame()

df = load_data()
countries = df['Country'].unique().tolist() if not df.empty else []
years = sorted(df['Year'].unique().tolist()) if not df.empty else []

def train_energy_model(df_p, n_estimators=100, random_state=42):
    X = df_p[['Country', 'Year']]
    y = df_p['Total_Energy_Consumption_EJ']

    # Preprocessing for Random Forest (Year needs to be treated numerically)
    preprocessor = ColumnTransformer(
        transformers=[
            ('country', OneHotEncoder(handle_unknown='ignore'), ['Country']),
            ('year', 'passthrough', ['Year'])
        ],
        remainder='passthrough'
    )

    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('regressor', RandomForestRegressor(n_estimators=n_estimators, random_state=random_state))
    ])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    pipeline.fit(X_train, y_train)
    return pipeline

def predict_future_energy(pipeline, country, year):
    prediction_df = pd.DataFrame([{'Country': country, 'Year': year}])
    prediction = pipeline.predict(prediction_df)
    return prediction[0]

df_p = pd.read_csv('Primary energy consumption - EJ.csv')
df_p = df_p.drop(columns=[col for col in df_p.columns if col.startswith('Unnamed:')])
rf_pipeline = train_energy_model(df_p)

def visualize_country_energy(country_name):
    country_data = df[df['Country'] == country_name].copy()
    if country_data.empty:
        return None
    fig, ax = plt.subplots(figsize=(12, 6))
    for fuel in country_data['Fuel type'].unique():
        subset = country_data[country_data['Fuel type'] == fuel]
        ax.plot(subset['Year'], subset['Energy Consumption EJ'], marker='o', linestyle='-', label=fuel)
    ax.set_xlabel('Year')
    ax.set_ylabel('Energy Consumption (EJ)')
    ax.legend()
    ax.grid(alpha=0.3)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    img = BytesIO()
    fig.savefig(img, format='png')
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode('utf8')
    plt.close(fig)
    return plot_url

def visualize_country_fuel(country_name, year):
    country_year_data = df[(df['Country'] == country_name) & (df['Year'] == year)].copy()
    if country_year_data.empty:
        return None
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(country_year_data['Fuel type'], country_year_data['Energy Consumption EJ'], color='skyblue')
    ax.set_xlabel('Fuel Type')
    ax.set_ylabel('Energy Consumption (EJ)')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    img = BytesIO()
    fig.savefig(img, format='png')
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode('utf8')
    plt.close(fig)
    return plot_url

def generate_base64_plot(fig):
    img = BytesIO()
    fig.savefig(img, format='png')
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode('utf8')
    plt.close(fig)
    return plot_url

@app.route('/', methods=['GET', 'POST'])
def index():
    multi_fuel_plot = None
    yearly_fuel_plot = None

    if request.method == 'POST':
        if 'multi_fuel_country' in request.form:
            selected_country_multi = request.form['multi_fuel_country']
            multi_fuel_plot = visualize_country_energy(selected_country_multi)

        if 'yearly_fuel_country' in request.form and 'year' in request.form:
            selected_country_yearly = request.form['yearly_fuel_country']
            selected_year = int(request.form['year'])
            yearly_fuel_plot = visualize_country_fuel(selected_country_yearly, selected_year)

    return render_template('index.html',
                           countries=countries,
                           years=years,
                           multi_fuel_plot=multi_fuel_plot,
                           yearly_fuel_plot=yearly_fuel_plot)

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    future_energy_prediction = None
    predicted_country = None
    predicted_year = None
    prediction_plot = None
    historical_plot = None
    comparison_plot = None

    if request.method == 'POST':
        if 'prediction_country' in request.form and 'prediction_year' in request.form:
            selected_country_pred = request.form['prediction_country']
            selected_year_pred = int(request.form['prediction_year'])

            if 1965 <= selected_year_pred <= 2023:
                flash("Prediction year must be after 2023.", "warning")
            else:
                future_energy_prediction = predict_future_energy(rf_pipeline, selected_country_pred, selected_year_pred)
                predicted_country = selected_country_pred
                predicted_year = selected_year_pred

                # 1. Create Prediction Bar Chart (Single Country)
                if future_energy_prediction is not None:
                    fig_pred, ax_pred = plt.subplots(figsize=(6, 4))
                    ax_pred.bar([f'{predicted_country} ({predicted_year})'], [future_energy_prediction], color='skyblue')
                    ax_pred.set_ylabel('Total Energy Consumption (EJ)')
                    ax_pred.set_title('Predicted Energy Consumption')
                    prediction_plot = generate_base64_plot(fig_pred)

                # 2. Create Historical Trend Line Chart (Single Country)
                country_history = df_p[df_p['Country'] == selected_country_pred].sort_values(by='Year')
                if not country_history.empty and future_energy_prediction is not None:
                    fig_hist, ax_hist = plt.subplots(figsize=(8, 5))
                    ax_hist.plot(country_history['Year'], country_history['Total_Energy_Consumption_EJ'], marker='o', linestyle='-', label='Historical Data')
                    ax_hist.scatter(selected_year_pred, future_energy_prediction, color='red', marker='x', s=100, label='Predicted Value')
                    ax_hist.set_xlabel('Year')
                    ax_hist.set_ylabel('Total Energy Consumption (EJ)')
                    ax_hist.set_title(f'Historical Trend and Prediction for {predicted_country}')
                    ax_hist.legend()
                    ax_hist.grid(True)
                    historical_plot = generate_base64_plot(fig_hist)

                # 3. Create Comparison Bar Chart of Predictions (for a fixed future year)
                predictions = {}
                for country in countries:
                    prediction = predict_future_energy(rf_pipeline, country, selected_year_pred)
                    if prediction is not None:
                        predictions[country] = prediction

                if predictions:
                    fig_comp, ax_comp = plt.subplots(figsize=(30, 10))
                    country_names = list(predictions.keys())
                    predicted_values = list(predictions.values())
                    ax_comp.bar(country_names, predicted_values, color='lightcoral')
                    ax_comp.set_xlabel('Country')
                    ax_comp.set_ylabel('Predicted Energy Consumption (EJ)')
                    ax_comp.set_title(f'Predicted Energy Consumption for {selected_year_pred} Across Countries')
                    ax_comp.tick_params(axis='x', rotation=45, labelbottom=True, labelleft=True)
                    plt.xticks(rotation=45, ha='right')
                    plt.tight_layout()
                    comparison_plot = generate_base64_plot(fig_comp)

    return render_template('predict.html',
                           countries=countries,
                           years=range(2024, 2101),
                           future_energy_prediction=future_energy_prediction,
                           predicted_country=predicted_country,
                           predicted_year=predicted_year,
                           prediction_plot=prediction_plot,
                           historical_plot=historical_plot,
                           comparison_plot=comparison_plot)

if __name__ == '__main__':
    app.run(debug=True)