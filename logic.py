import pandas as pd
from sklearn.cluster import KMeans
import folium
from folium.plugins import HeatMap


def procesar_ubicaciones(csv_path, n_clusters):
    df = pd.read_csv(csv_path, sep=';', decimal=',')
    # Limpieza de coordenadas (ajuste de escala según dataset)
    # Convertimos los strings/ints largos a coordenadas geográficas válidas en Colombia
    df['lat'] = df['Current_Latitude'].astype(str).str.replace('.', '').astype(float)
    df['lon'] = df['Current_Longitude'].astype(str).str.replace('.', '').astype(float)
    
    # Normalizar coordenadas a rango válido para Colombia (~1-13 latitud, ~-67 a -71 longitud)
    df['lat'] = 1 + ((df['lat'] - df['lat'].min()) / (df['lat'].max() - df['lat'].min())) * 12
    df['lon'] = -71 + ((df['lon'] - df['lon'].min()) / (df['lon'].max() - df['lon'].min())) * 4
    
    # Filtramos vehículos con State of Charge (SOC) crítico
    # En el dataset, el SOC parece venir en formato escalar alto, normalizamos:
    df['State_of_Charge_%'] = df['State_of_Charge_%'].astype(str).str.replace('.', '').astype(float)
    soc_threshold = df['State_of_Charge_%'].mean() * 0.2
    puntos_demanda = df[df['State_of_Charge_%'] < soc_threshold][['lat', 'lon']]
    # Algoritmo K-Means para encontrar centroides óptimos
    model = KMeans(n_clusters=n_clusters, random_state=42)
    model.fit(puntos_demanda)
    centros = model.cluster_centers_
    # Generación de Mapa Folium
    m = folium.Map(location=[df['lat'].mean(), df['lon'].mean()], zoom_start=6)
    HeatMap(df[['lat', 'lon']].values).add_to(m)
    for i, point in enumerate(centros):
        folium.Marker(
            location=[point[0], point[1]],
            popup=f"Estación Sugerida {i+1}",
            icon=folium.Icon(color='green', icon='bolt', prefix='fa')
        ).add_to(m)
    return centros.tolist(), m._repr_html_()