import os
import sqlite3
import numpy as np
from flask import Flask, render_template, request, session, redirect, url_for, flash
from tensorflow.keras.models import load_model
from joblib import load
import mne
import pandas as pd
from werkzeug.security import generate_password_hash, check_password_hash
from functools import wraps
from datetime import datetime

app = Flask(__name__)
app.secret_key = 'your-secret-key-change-this-in-production'  # IMPORTANT: Change this!

# Database config
DATABASE = 'sleep_monitor.db'

# Paths
MODEL_PATH = os.path.join("models", "sleep_cnn_lstm_best.h5")
LIFESTYLE_MODEL_PATH = os.path.join("models", "sleep_disorder_hybrid.joblib")
LIFESTYLE_ENCODERS_PATH = os.path.join("models", "lifestyle_encoders.joblib")
DISORDER_ENCODER_PATH = os.path.join("models", "disorder_encoder.joblib")
LIFESTYLE_SCALER_PATH = os.path.join("models", "lifestyle_scaler.joblib")
UPLOAD_FOLDER = os.path.join("uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ===============================
# DATABASE FUNCTIONS
# ===============================
def get_db():
    """Get database connection"""
    db = sqlite3.connect(DATABASE)
    db.row_factory = sqlite3.Row
    return db

def init_db():
    """Initialize SQLite database with users table"""
    db = get_db()
    db.execute('''
        CREATE TABLE IF NOT EXISTS users (
            user_id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            email TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            full_name TEXT,
            age INTEGER,
            gender TEXT CHECK(gender IN ('Male', 'Female', 'Other')),
            phone TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            last_login TIMESTAMP,
            is_active INTEGER DEFAULT 1
        )
    ''')
    db.execute('CREATE INDEX IF NOT EXISTS idx_email ON users(email)')
    db.execute('CREATE INDEX IF NOT EXISTS idx_username ON users(username)')
    db.commit()
    db.close()
    print("Database initialized!")

# Initialize database on startup
init_db()

# Load ML models
print("Loading PSG Sleep Stage model...")
sleep_model = load_model(MODEL_PATH)

print("Loading Lifestyle Disorder model...")
lifestyle_model = load(LIFESTYLE_MODEL_PATH)
lifestyle_encoders = load(LIFESTYLE_ENCODERS_PATH)
disorder_encoder = load(DISORDER_ENCODER_PATH)
lifestyle_scaler = load(LIFESTYLE_SCALER_PATH)

# Extract feature importance
rf_estimator = lifestyle_model.named_estimators_['rf']
feature_cols = [
    'Gender', 'Age', 'Occupation',
    'Sleep Duration (hours)', 'Quality of Sleep (scale: 1-10)',
    'Physical Activity Level (minutes/day)', 'Stress Level (scale: 1-10)',
    'BMI Category', 'Blood Pressure', 'Heart Rate (bpm)', 'Daily Steps'
]
feature_importance = dict(zip(feature_cols, rf_estimator.feature_importances_))
top_reasons = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:5]

app.config['FEATURE_COLS'] = feature_cols
app.config['TOP_REASONS'] = top_reasons
print("All models loaded!")

STAGE_NAMES = ['W', 'N1', 'N2', 'N3', 'REM']

# ===============================
# DECORATORS
# ===============================
def login_required(f):
    """Decorator to require login for protected routes"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            flash('Please log in to access this page.', 'warning')
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

# ===============================
# AUTHENTICATION ROUTES
# ===============================
@app.route('/')
def index():
    """Landing page"""
    if 'user_id' in session:
        return redirect(url_for('dashboard'))
    return render_template('index.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    """User registration with SQLite"""
    if request.method == 'POST':
        username = request.form.get('username')
        email = request.form.get('email')
        password = request.form.get('password')
        confirm_password = request.form.get('confirm_password')
        full_name = request.form.get('full_name')
        age = request.form.get('age')
        gender = request.form.get('gender')
        phone = request.form.get('phone')

        # Validation
        if not all([username, email, password, confirm_password]):
            flash('Please fill in all required fields.', 'error')
            return render_template('register.html')

        if password != confirm_password:
            flash('Passwords do not match.', 'error')
            return render_template('register.html')

        if len(password) < 6:
            flash('Password must be at least 6 characters long.', 'error')
            return render_template('register.html')

        try:
            db = get_db()
            cursor = db.cursor()

            # Check if username or email already exists
            cursor.execute("SELECT user_id FROM users WHERE username = ? OR email = ?", (username, email))
            if cursor.fetchone():
                flash('Username or email already exists.', 'error')
                db.close()
                return render_template('register.html')

            # Hash password and insert user
            password_hash = generate_password_hash(password)
            cursor.execute("""
                INSERT INTO users (username, email, password_hash, full_name, age, gender, phone)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (username, email, password_hash, full_name, age, gender, phone))

            db.commit()
            db.close()

            flash('Registration successful! Please log in.', 'success')
            return redirect(url_for('login'))

        except Exception as e:
            flash(f'An error occurred during registration: {str(e)}', 'error')
            return render_template('register.html')

    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    """User login with SQLite"""
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')

        if not username or not password:
            flash('Please enter both username and password.', 'error')
            return render_template('login.html')

        try:
            db = get_db()
            cursor = db.cursor()
            cursor.execute("""
                SELECT user_id, username, email, password_hash, full_name, is_active
                FROM users WHERE username = ? OR email = ?
            """, (username, username))

            user = cursor.fetchone()

            if user and check_password_hash(user['password_hash'], password):
                if not user['is_active']:
                    flash('Your account has been deactivated. Please contact support.', 'error')
                    db.close()
                    return render_template('login.html')

                # Set session
                session['user_id'] = user['user_id']
                session['username'] = user['username']
                session['email'] = user['email']
                session['full_name'] = user['full_name']

                # Update last login
                cursor.execute("UPDATE users SET last_login = ? WHERE user_id = ?",
                             (datetime.now(), user['user_id']))
                db.commit()
                db.close()

                flash(f'Welcome back, {user["full_name"] or user["username"]}!', 'success')
                return redirect(url_for('dashboard'))
            else:
                flash('Invalid username or password.', 'error')
                db.close()
                return render_template('login.html')

        except Exception as e:
            flash(f'An error occurred during login: {str(e)}', 'error')
            return render_template('login.html')

    return render_template('login.html')

@app.route('/logout')
@login_required
def logout():
    """User logout"""
    session.clear()
    flash('You have been logged out successfully.', 'info')
    return redirect(url_for('index'))


@app.route('/profile', methods=['GET', 'POST'])
@login_required
def profile():
    """User profile page with edit functionality"""
    db = get_db()
    cursor = db.cursor()

    if request.method == 'POST':
        action = request.form.get('action')

        # UPDATE PROFILE
        if action == 'update_profile':
            full_name = request.form.get('full_name')
            age = request.form.get('age')
            gender = request.form.get('gender')
            phone = request.form.get('phone')
            email = request.form.get('email')

            try:
                # Check if email is already taken by another user
                cursor.execute("SELECT user_id FROM users WHERE email = ? AND user_id != ?",
                               (email, session['user_id']))
                if cursor.fetchone():
                    flash('Email is already in use by another account.', 'error')
                    db.close()
                    return redirect(url_for('profile'))

                cursor.execute("""
                    UPDATE users 
                    SET full_name = ?, age = ?, gender = ?, phone = ?, email = ?
                    WHERE user_id = ?
                """, (full_name, age, gender, phone, email, session['user_id']))

                db.commit()
                session['email'] = email
                session['full_name'] = full_name
                flash('Profile updated successfully!', 'success')

            except Exception as e:
                flash(f'Error updating profile: {str(e)}', 'error')
            finally:
                db.close()

            return redirect(url_for('profile'))

        # CHANGE PASSWORD
        elif action == 'change_password':
            current_password = request.form.get('current_password')
            new_password = request.form.get('new_password')
            confirm_password = request.form.get('confirm_password')

            if not all([current_password, new_password, confirm_password]):
                flash('Please fill in all password fields.', 'error')
                db.close()
                return redirect(url_for('profile'))

            if new_password != confirm_password:
                flash('New passwords do not match.', 'error')
                db.close()
                return redirect(url_for('profile'))

            if len(new_password) < 6:
                flash('Password must be at least 6 characters long.', 'error')
                db.close()
                return redirect(url_for('profile'))

            try:
                cursor.execute("SELECT password_hash FROM users WHERE user_id = ?",
                               (session['user_id'],))
                user = cursor.fetchone()

                if not check_password_hash(user['password_hash'], current_password):
                    flash('Current password is incorrect.', 'error')
                    db.close()
                    return redirect(url_for('profile'))

                new_password_hash = generate_password_hash(new_password)
                cursor.execute("UPDATE users SET password_hash = ? WHERE user_id = ?",
                               (new_password_hash, session['user_id']))
                db.commit()
                flash('Password changed successfully!', 'success')

            except Exception as e:
                flash(f'Error changing password: {str(e)}', 'error')
            finally:
                db.close()

            return redirect(url_for('profile'))

    # GET request - display profile
    cursor.execute("SELECT * FROM users WHERE user_id = ?", (session['user_id'],))
    user = cursor.fetchone()
    db.close()
    return render_template('profile.html', user=user)


# ===============================
# ML PREDICTION FUNCTIONS
# ===============================
def preprocess_psg_for_prediction(psg_path):
    """Extract 30s EEG epochs for sleep staging"""
    raw = mne.io.read_raw_edf(psg_path, preload=True, verbose=False)
    raw.resample(100)

    if "Fpz-Cz" in raw.ch_names:
        picks = mne.pick_channels(raw.ch_names, include=["Fpz-Cz"])
    else:
        picks = mne.pick_types(raw.info, eeg=True)

    if len(picks) == 0:
        raise ValueError("No EEG channels found in PSG file")

    eeg_data = raw.get_data(picks=picks)[0]
    sfreq = raw.info['sfreq']
    n_samples_epoch = int(30 * sfreq)
    n_epochs = len(eeg_data) // n_samples_epoch

    if n_epochs == 0:
        raise ValueError("PSG file too short for 30s epochs")

    epochs_data = eeg_data[:n_epochs * n_samples_epoch].reshape(n_epochs, n_samples_epoch)
    epochs_data = (epochs_data - epochs_data.mean(axis=1, keepdims=True)) / \
                  (epochs_data.std(axis=1, keepdims=True) + 1e-6)
    epochs_data = epochs_data[..., np.newaxis]
    return epochs_data

def predict_sleep_disorder_risk(form_data):
    """Predict sleep disorder risk + contributing factors"""
    X_lifestyle = pd.DataFrame([form_data])

    for col, le in lifestyle_encoders.items():
        X_lifestyle[col] = le.transform(X_lifestyle[col].astype(str))

    num_cols = ['Age', 'Sleep Duration (hours)', 'Quality of Sleep (scale: 1-10)',
                'Physical Activity Level (minutes/day)', 'Stress Level (scale: 1-10)',
                'Blood Pressure', 'Heart Rate (bpm)', 'Daily Steps']
    X_lifestyle[num_cols] = lifestyle_scaler.transform(X_lifestyle[num_cols])

    pred_idx = lifestyle_model.predict(X_lifestyle)[0]
    pred_proba = np.max(lifestyle_model.predict_proba(X_lifestyle)[0])

    contrib_features = {k: v for k, v in form_data.items()}
    risk = disorder_encoder.inverse_transform([pred_idx])[0]
    return risk, f"{pred_proba * 100:.1f}%", contrib_features

# ===============================
# DASHBOARD ROUTE (Protected)
# ===============================
@app.route("/dashboard", methods=["GET", "POST"])
@login_required
def dashboard():
    """Main dashboard - requires login"""
    results = None
    error = None

    if request.method == "POST":
        try:
            psg_results = {}
            risk_results = {}

            # PSG Sleep Stage Analysis
            psg_file = request.files.get("psg_file")
            if psg_file and psg_file.filename:
                filename = os.path.join(UPLOAD_FOLDER, psg_file.filename)
                psg_file.save(filename)

                X_test = preprocess_psg_for_prediction(filename)
                predictions = sleep_model.predict(X_test, verbose=0)
                pred_classes = np.argmax(predictions, axis=1)
                pred_labels = [STAGE_NAMES[i] for i in pred_classes]

                psg_results = {
                    "filename": psg_file.filename,
                    "dominant_stage": max(set(pred_labels), key=pred_labels.count),
                    "total_epochs": len(pred_labels),
                    "stage_distribution": {stage: pred_labels.count(stage) for stage in STAGE_NAMES}
                }
                os.remove(filename)

            # Lifestyle Risk Prediction
            lifestyle_data = {
                'Gender': request.form.get('gender', 'Male'),
                'Age': float(request.form.get('age', 30)),
                'Occupation': request.form.get('occupation', 'Office Worker'),
                'Sleep Duration (hours)': float(request.form.get('sleep_hours', 7)),
                'Quality of Sleep (scale: 1-10)': float(request.form.get('sleep_quality', 7)),
                'Physical Activity Level (minutes/day)': float(request.form.get('activity_min', 30)),
                'Stress Level (scale: 1-10)': float(request.form.get('stress', 5)),
                'BMI Category': request.form.get('bmi_category', 'Normal'),
                'Blood Pressure': float(request.form.get('blood_pressure', 120)),
                'Heart Rate (bpm)': float(request.form.get('heart_rate', 72)),
                'Daily Steps': float(request.form.get('daily_steps', 8000))
            }

            risk, risk_prob, contrib_features = predict_sleep_disorder_risk(lifestyle_data)
            risk_results = {
                "risk": risk,
                "risk_prob": risk_prob,
                "contributing_factors": contrib_features,
                "top_global_reasons": app.config['TOP_REASONS']
            }

            form_data_summary = {
                'sleep_hours': lifestyle_data['Sleep Duration (hours)'],
                'sleep_quality': lifestyle_data['Quality of Sleep (scale: 1-10)'],
                'activity_min': lifestyle_data['Physical Activity Level (minutes/day)'],
                'stress': lifestyle_data['Stress Level (scale: 1-10)']
            }

            results = {
                "psg": psg_results if psg_results else None,
                "risk": risk_results,
                "form_data": form_data_summary
            }

        except Exception as e:
            error = f"Analysis error: {str(e)}"

    return render_template("dashboard.html", results=results, error=error)

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
