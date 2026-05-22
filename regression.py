import pandas as pd
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.tree import DecisionTreeRegressor

CSV_PATH = "cvs/dataset.csv"
PREDICTION_PLOT_PATH = "static/regression_predictions.png"
RESIDUAL_PLOT_PATH = "static/regression_residuals.png"
FEATURE_COLUMNS = [
    "Battery_Capacity_kWh",
    "State_of_Charge_%",
    "Energy_Consumption_Rate_kWh/km",
    "Distance_to_Destination_km",
    "Traffic_Data",
    "Charging_Rate_kW",
    "Queue_Time_mins",
    "Station_Capacity_EV",
    "Time_Spent_Charging_mins",
    "Session_Start_Hour",
    "Fleet_Size",
    "Temperature_C",
    "Wind_Speed_m/s",
    "Precipitation_mm",
    "Weekday"
]
FEATURE_LABELS = {
    "Battery_Capacity_kWh": "Battery Capacity (kWh)",
    "State_of_Charge_%": "State of Charge (%)",
    "Energy_Consumption_Rate_kWh/km": "Energy Consumption Rate (kWh/km)",
    "Distance_to_Destination_km": "Distance to Destination (km)",
    "Traffic_Data": "Traffic Data",
    "Charging_Rate_kW": "Charging Rate (kW)",
    "Queue_Time_mins": "Queue Time (mins)",
    "Station_Capacity_EV": "Station Capacity (EV)",
    "Time_Spent_Charging_mins": "Time Spent Charging (mins)",
    "Session_Start_Hour": "Session Start Hour",
    "Fleet_Size": "Fleet Size",
    "Temperature_C": "Temperature (C)",
    "Wind_Speed_m/s": "Wind Speed (m/s)",
    "Precipitation_mm": "Precipitation (mm)",
    "Weekday": "Weekday"
}


def clean_number(value):
    value = str(value)

    if value.count(".") > 1:
        parts = value.split(".")
        value = parts[0] + "." + "".join(parts[1:])

    try:
        return float(value)
    except ValueError:
        return None


def create_regression_plots(y_test, predictions):
    os.makedirs("static", exist_ok=True)

    plot_data = pd.DataFrame({
        "actual": y_test,
        "predicted": predictions
    }).sample(n=min(300, len(y_test)), random_state=42)

    plt.figure(figsize=(8, 5))
    plt.scatter(plot_data["actual"], plot_data["predicted"], alpha=0.65, color="#0d6efd")
    min_value = min(plot_data["actual"].min(), plot_data["predicted"].min())
    max_value = max(plot_data["actual"].max(), plot_data["predicted"].max())
    plt.plot([min_value, max_value], [min_value, max_value], color="#dc3545", linewidth=2)
    plt.title("Actual vs Predicted Energy Drawn")
    plt.xlabel("Actual Energy Drawn (kWh)")
    plt.ylabel("Predicted Energy Drawn (kWh)")
    plt.tight_layout()
    plt.savefig(PREDICTION_PLOT_PATH)
    plt.close()

    residuals = y_test - predictions
    residual_sample = pd.Series(residuals).sample(n=min(300, len(residuals)), random_state=42)

    plt.figure(figsize=(8, 5))
    plt.scatter(range(len(residual_sample)), residual_sample, alpha=0.65, color="#198754")
    plt.axhline(y=0, color="#dc3545", linewidth=2)
    plt.title("Regression Residual Errors")
    plt.xlabel("Sample")
    plt.ylabel("Actual - Predicted")
    plt.tight_layout()
    plt.savefig(RESIDUAL_PLOT_PATH)
    plt.close()


def evaluate_regression_models(X_train, X_test, y_train, y_test):
    models = {
        "Linear Regression": LinearRegression(),
        "Decision Tree Regressor": DecisionTreeRegressor(
            max_depth=10,
            random_state=42
        ),
        "Random Forest Regressor": RandomForestRegressor(
            n_estimators=50,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        ),
        "Gradient Boosting Regressor": GradientBoostingRegressor(
            random_state=42
        )
    }

    comparison = []
    trained_models = {}

    for model_name, model in models.items():
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)

        comparison.append({
            "name": model_name,
            "r2_score": round(r2_score(y_test, predictions), 4),
            "mean_absolute_error": round(mean_absolute_error(y_test, predictions), 4),
            "root_mean_squared_error": float(round(np.sqrt(mean_squared_error(y_test, predictions)), 4))
        })

        trained_models[model_name] = {
            "model": model,
            "predictions": predictions
        }

    comparison = sorted(
        comparison,
        key=lambda item: item["mean_absolute_error"]
    )

    best_model_name = comparison[0]["name"]

    for item in comparison:
        item["is_best"] = item["name"] == best_model_name

    return comparison, trained_models, best_model_name


def build_prediction_inputs(df):
    inputs = []

    for column in FEATURE_COLUMNS:
        inputs.append({
            "name": column,
            "label": FEATURE_LABELS[column],
            "value": round(float(df[column].median()), 4)
        })

    return inputs


def predict_energy_drawn(model, form_data):
    input_values = {}

    for column in FEATURE_COLUMNS:
        value = clean_number(form_data.get(column))

        if value is None:
            raise ValueError(f"Invalid value for {FEATURE_LABELS[column]}")

        input_values[column] = value

    input_df = pd.DataFrame([input_values], columns=FEATURE_COLUMNS)
    prediction = model.predict(input_df)[0]

    return round(float(prediction), 4), input_values


def train_regression_model():
    df = pd.read_csv(CSV_PATH, sep=";")

    target_column = "Energy_Drawn_kWh"

    numeric_columns = FEATURE_COLUMNS + [target_column]

    original_records = len(df)

    for column in numeric_columns:
        df[column] = df[column].apply(clean_number)

    df = df.dropna(subset=numeric_columns)

    clean_records = len(df)
    removed_records = original_records - clean_records

    X = df[FEATURE_COLUMNS]
    y = df[target_column]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    model_comparison, trained_models, best_model_name = evaluate_regression_models(
        X_train,
        X_test,
        y_train,
        y_test
    )

    model = trained_models["Linear Regression"]["model"]
    predictions = trained_models["Linear Regression"]["predictions"]
    create_regression_plots(y_test, predictions)

    r2 = round(r2_score(y_test, predictions), 4)

    if r2 == -0.0:
        r2 = 0.0

    results = {
        "target": target_column,
        "original_records": original_records,
        "clean_records": clean_records,
        "removed_records": removed_records,
        "r2_score": r2,
        "mean_absolute_error": round(mean_absolute_error(y_test, predictions), 4),
        "best_model": best_model_name,
        "model_comparison": model_comparison,
        "feature_inputs": build_prediction_inputs(df),
        "prediction_plot": "regression_predictions.png",
        "residual_plot": "regression_residuals.png"
    }

    return model, results
