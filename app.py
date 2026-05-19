from flask import Flask, render_template, request
import logic
import os

app = Flask(__name__)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)

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

if __name__ == '__main__':
    app.run(debug=True)