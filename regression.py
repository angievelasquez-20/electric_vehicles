import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error

CSV_PATH = "cvs/dataset.csv"


def clean_number(value):
    value = str(value)

    if value.count(".") > 1:
        parts = value.split(".")
        value = parts[0] + "." + "".join(parts[1:])

    try:
        return float(value)
    except ValueError:
        return None


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

    r2 = round(r2_score(y_test, predictions), 4)

    if r2 == -0.0:
        r2 = 0.0

    results = {
        "target": target_column,
        "original_records": original_records,
        "clean_records": clean_records,
        "removed_records": removed_records,
        "r2_score": r2,
        "mean_absolute_error": round(mean_absolute_error(y_test, predictions), 4)
    }

    return model, results