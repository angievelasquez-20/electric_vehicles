from flask import Flask, render_template, request
import logic
from regression import train_regression_model
import os
import kmeans as Clustering

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/index')
def index():
    return render_template('index.html')

@app.route('/presentation')
def presentation():
    return render_template('presentation.html')

@app.route('/Recommend', methods=['GET', 'POST'])
def Recommend():
    if request.method == 'POST':
        try:
            n_estaciones = int(request.form.get('n_estaciones', 5))
            estaciones, mapa_html = logic.procesar_ubicaciones('cvs/dataset.csv', n_estaciones)
            return render_template('map.html', mapa=mapa_html, estaciones=estaciones)
        except Exception as e:
            return f"Error: {str(e)}", 500
    return render_template('index.html')

@app.route('/regression')
def regression():
    try:
        model, results = train_regression_model()

        return render_template(
            'regression.html',
            results=results
        )

    except Exception as e:
        return f"Linear regression error: {str(e)}", 500


@app.route('/K_means', methods=['GET', 'POST'])
def K_means():

    results = None
    summaryClusters = None
    centers = None
    plot_url = None

    if request.method == 'POST':

        k = int(request.form['k'])

        info = Clustering.applyClusteringKmeans(k)

        results = info["results"]
        summaryClusters = info["summaryClusters"]
        centers = info["centers"]

        Clustering.generate_plot(results, centers)

        plot_url = 'kmeans_plot.png'

    return render_template(
        'k_means.html',
        results=results,
        summaryClusters=summaryClusters,
        centers=centers,
        plot_url=plot_url
    )


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
