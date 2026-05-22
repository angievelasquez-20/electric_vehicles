import pandas as pd
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error

CSV_PATH = "cvs/dataset.csv"
PREDICTION_PLOT_PATH = "static/regression_predictions.png"
RESIDUAL_PLOT_PATH = "static/regression_residuals.png"


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


def train_regression_model():
    df = pd.read_csv(CSV_PATH, sep=";")

    feature_columns = [
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

    target_column = "Energy_Drawn_kWh"

    numeric_columns = feature_columns + [target_column]

    original_records = len(df)

    for column in numeric_columns:
        df[column] = df[column].apply(clean_number)

    df = df.dropna(subset=numeric_columns)

    clean_records = len(df)
    removed_records = original_records - clean_records

    X = df[feature_columns]
    y = df[target_column]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    model = LinearRegression()
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
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
        "prediction_plot": "regression_predictions.png",
        "residual_plot": "regression_residuals.png"
    }

    return model, results
