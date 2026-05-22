import pandas as pd
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


def applyClusteringKmeans(k):

    df = pd.read_csv(
        'cvs/dataset.csv',
        sep=';',
        nrows=10000
    )

    df.columns = df.columns.str.strip()

    columns_to_clean = [
        "Current_Latitude",
        "Current_Longitude",
        "Traffic_Data",
        "Charging_Load_kW"
    ]

    for col in columns_to_clean:

        df[col] = (
            df[col]
            .astype(str)
            .str.replace('.', '', regex=False)
            .str.replace(',', '.', regex=False)
        )

        df[col] = pd.to_numeric(
            df[col],
            errors='coerce'
        )

    df = df.dropna(subset=columns_to_clean)

    X = df[[
        "Current_Latitude",
        "Current_Longitude",
        "Traffic_Data",
        "Charging_Load_kW"
    ]].values

    scaler = StandardScaler()

    X_scaled = scaler.fit_transform(X)

    model = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=5
    )

    labels = model.fit_predict(X_scaled)

    df["cluster"] = labels

    results = df.to_dict(orient="records")

    summaryClusters = (
        df["cluster"]
        .value_counts()
        .to_dict()
    )

    centers = scaler.inverse_transform(
        model.cluster_centers_
    ).tolist()

    return {
        "results": results,
        "summaryClusters": summaryClusters,
        "centers": centers
    }


def generate_plot(results, centers):

    limited_results = results[:50000]

    x = [
        row["Current_Longitude"]
        for row in limited_results
    ]

    y = [
        row["Current_Latitude"]
        for row in limited_results
    ]

    clusters = [
        row["cluster"]
        for row in limited_results
    ]

    plt.figure(figsize=(6, 4))

    plt.scatter(
        x,
        y,
        c=clusters,
        cmap='viridis',
        s=20
    )

    cx = [c[1] for c in centers]

    cy = [c[0] for c in centers]

    plt.scatter(
        cx,
        cy,
        color='red',
        marker='X',
        s=250,
        edgecolors='black',
        linewidths=2
    )

    for i in range(len(cx)):

        plt.text(
            cx[i] + 0.02,
            cy[i] + 0.02,
            f'C{i}',
            fontsize=9,
            color='black',
            weight='bold'
        )

    plt.xlabel("Longitude")

    plt.ylabel("Latitude")

    plt.title(
        "K-Means Clustering - EV Charging Stations"
    )

    plt.tight_layout()

    plt.savefig(
        'static/kmeans_plot.png'
    )

    plt.close('all')