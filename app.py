from flask import Flask, render_template, request
import logic
from regression import train_regression_model

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/index')
def index():
    return render_template('index.html')

@app.route('/recomendar', methods=['GET', 'POST'])
def recomendar():
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

if __name__ == '__main__':
    app.run(debug=True)