from flask import Flask, render_template, request
import logic
from regression import train_regression_model
import os
import atexit
import shutil
import signal
import auth_ad
from flask import session, redirect, url_for, flash
from flask_session import Session as FlaskSession

app = Flask(__name__)
app.secret_key = 'tu_clave_secreta_para_sesiones' # Required for using sessions

# Configure server-side sessions (filesystem). This lets the app clear
# session files when the program exits so sessions don't persist after shutdown.
SESSION_DIR = os.path.join(os.path.dirname(__file__), 'session_files')
app.config['SESSION_TYPE'] = 'filesystem'
app.config['SESSION_FILE_DIR'] = SESSION_DIR
app.config['SESSION_PERMANENT'] = False
FlaskSession(app)

# Ensure session files are removed when the program terminates
def _clear_session_files():
    try:
        if os.path.isdir(SESSION_DIR):
            shutil.rmtree(SESSION_DIR)
    except Exception:
        pass

# Clear stale session files when the app starts
_clear_session_files()
os.makedirs(SESSION_DIR, exist_ok=True)

atexit.register(_clear_session_files)

# Handle normal termination signals in addition to atexit
def _shutdown_handler(signum, frame):
    _clear_session_files()
    raise SystemExit(0)

for _signal in (signal.SIGINT, signal.SIGTERM):
    signal.signal(_signal, _shutdown_handler)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/index')
def index():
    return render_template('index.html')

@app.route('/presentation')
def presentation():
    return render_template('presentation.html')

@app.route('/recommend', methods=['GET', 'POST'])
def recommend():
    if request.method == 'POST':
        try:
            n_stations = int(request.form.get('n_stations', 5))
            stations, map_html = logic.process_locations('cvs/dataset.csv', n_stations)
            return render_template('map.html', map=map_html, stations=stations)
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
    

@app.route('/login')
def login():
    # If already logged in, go directly to the optimizer
    if 'user' in session:
        return redirect(url_for('optimizer'))
    return render_template('login.html')

@app.route('/login_process', methods=['POST'])
def login_process():
    username = request.form.get('username')
    password = request.form.get('password')
    
    # Validate with Windows Server Active Directory
    if auth_ad.authenticate_ad_user(username, password):
        session.permanent = False
        session['user'] = username  # store the session
        return redirect(url_for('optimizer'))
    else:
        flash('Active Directory credentials are incorrect or server unavailable.', 'danger')
        return redirect(url_for('login'))


@app.route('/optimizer')
def optimizer():
    # Protect route: if no session, redirect to login
    if 'user' not in session:
        return redirect(url_for('login'))
    return render_template('optimizer.html')


@app.route('/logout')
def logout():
    session.pop('user', None)
    return redirect(url_for('login'))


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
