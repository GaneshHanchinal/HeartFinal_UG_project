import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import sqlite_utils
import bcrypt
import sqlite3 
from sqlite_utils.db import NotFoundError

# NEW IMPORTS FOR SELF-CONTAINED MODEL SETUP
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
from imblearn.pipeline import Pipeline 

# --- 1. CONFIGURATION AND INITIALIZATION ---

st.set_page_config(
    page_title="Secure Heart Disease Predictor",
    page_icon="❤️",
    layout="wide"
)

# File names (Must match files in GitHub repo)
MODEL_FILE = 'logistic_model.pkl'
DB_PATH = 'users.db'
DATA_FILE = 'heart.csv' # Added: Critical for self-training on cloud

# Session State Initialization
if 'logged_in' not in st.session_state:
    st.session_state['logged_in'] = False
if 'menu_selection' not in st.session_state:
    st.session_state['menu_selection'] = 'Login'
if 'name' not in st.session_state:
    st.session_state['name'] = None

# --- 2. MODEL AND DATABASE ACCESS FUNCTIONS (SELF-SUFFICIENT SETUP) ---

def initialize_database(db):
    """Ensures the 'patients' table exists."""
    try:
        # Create table if it does not exist
        db["patients"].create({
            "username": str,
            "password_hash": str,
            "name": str
        }, pk="username", if_not_exists=True)
    except Exception as e:
        # Handle cases where the table creation still fails
        st.error(f"FATAL DB ERROR: Could not ensure 'patients' table exists: {e}")
        return False
    return True

@st.cache_resource
def get_db():
    """
    Retrieves the database connection. Creates the DB file and table if missing.
    CRITICAL: uses check_same_thread=False for Streamlit compatibility.
    """
    try:
        # If the file doesn't exist, sqlite3.connect CREATES IT (solving the cloud issue).
        conn = sqlite3.connect(DB_PATH, check_same_thread=False)
        db = sqlite_utils.Database(conn)
        
        # Ensure the table is created
        if not initialize_database(db):
            return None
            
        return db
    except Exception as e:
        st.error(f"FATAL ERROR connecting to database: {e}")
        return None 


@st.cache_resource(show_spinner="Training model on first run...")
def run_setup():
    """
    Consolidated function to train the model and save it locally 
    if it doesn't exist, making the app self-sufficient for deployment.
    """
    if os.path.exists(MODEL_FILE):
        try:
            with open(MODEL_FILE, 'rb') as file:
                model = pickle.load(file)
            return model
        except Exception:
            # If the file exists but is corrupted, we re-train
            os.remove(MODEL_FILE) 

    # --- MODEL TRAINING LOGIC (Adapted from final_setup.py) ---
    st.info("Model file not found. Training robust model now (this may take a moment)...")
    
    # CRITICAL CHECK: Ensure data file is present on the cloud
    if not os.path.exists(DATA_FILE):
        st.error(f"❌ FATAL ERROR: Data file '{DATA_FILE}' not found. Ensure it's in your GitHub repo.")
        st.stop()
        
    try:
        df = pd.read_csv(DATA_FILE)
    except Exception as e:
        st.error(f"❌ FATAL ERROR reading {DATA_FILE}: {e}")
        st.stop()
        
    FEATURES = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 
                'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']
    TARGET = 'target'
    
    X = df[FEATURES].fillna(df[FEATURES].mean())
    y = df[TARGET]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Define the Robust Pipeline
    steps = [
        ('scaler', StandardScaler()),                 # Step 1: Feature Scaling
        ('smote', SMOTE(random_state=42)),           # Step 2: Data Balancing
        ('logreg', LogisticRegression(
            max_iter=5000, 
            solver='liblinear', 
            penalty='l2', 
            C=0.5, 
            random_state=42
        ))                                           # Step 3: Regularized Model
    ]
    pipeline = Pipeline(steps=steps)

    # Train the Pipeline
    pipeline.fit(X_train, y_train)
    
    # Save the trained pipeline model
    with open(MODEL_FILE, 'wb') as file:
        pickle.dump(pipeline, file)
    
    # NOTE: Accuracy check removed from this function to keep the logic clean, 
    # but the model is saved and returned.
    
    st.success("Model training complete!")
    return pipeline

# --- 3. USER AUTHENTICATION FUNCTIONS ---

def hash_password(password):
    """Hashes a password using bcrypt."""
    salt = bcrypt.gensalt()
    return bcrypt.hashpw(password.encode('utf-8'), salt)

def check_password(password, hashed_password):
    """Checks a plain password against a hashed password."""
    try:
        return bcrypt.checkpw(password.encode('utf-8'), hashed_password.encode('utf-8'))
    except ValueError:
        return False

def register_user(username, password, name):
    """Adds a new user to the database using a robust method."""
    db = get_db()
    if db is None:
        return False, "Database connection is unavailable."

    try:
        # Robustly check if user exists (using list() for compatibility)
        existing_users = list(db.query(
            "SELECT username FROM patients WHERE username = :username", 
            {"username": username}
        ))
        
        if existing_users:
            return False, "Username already exists."
        
        hashed = hash_password(password)
        
        db["patients"].insert({
            "username": username,
            "password_hash": hashed.decode('utf-8'), 
            "name": name
        }, pk="username")
        return True, "Registration successful! Please log in."
    except Exception as e:
        return False, f"Database error during registration: {e}"

def verify_login(username, password):
    """Verifies a user's credentials against the database using a robust method."""
    db = get_db()
    if db is None:
        return False, None
        
    try:
        # Robustly fetch user data (using list() for compatibility)
        user_data_list = list(db.query(
            "SELECT * FROM patients WHERE username = :username", 
            {"username": username}
        ))
    except Exception:
        return False, None
    
    if user_data_list:
        user_data = user_data_list[0]
        if check_password(password, user_data["password_hash"]):
            return True, user_data["name"]
    
    return False, None

def logout():
    """Logs out the user and clears session state."""
    st.session_state['logged_in'] = False
    st.session_state['name'] = None
    st.info("Logged out successfully.")
    st.rerun()

# --- 4. LOGIN/REGISTRATION FORM DISPLAY ---

def show_registration_form():
    """Displays the new user registration form."""
    st.subheader("New Patient Registration ✍️")
    
    with st.form("registration_form"):
        new_name = st.text_input("Full Name")
        new_username = st.text_input("Choose Username (unique)")
        new_password = st.text_input("Set Password", type="password")
        confirm_password = st.text_input("Confirm Password", type="password")
        
        submitted = st.form_submit_button("Register")
        
        if submitted:
            if not all([new_username, new_password, new_name]):
                st.warning("All fields are required.")
            elif new_password != confirm_password:
                st.error("Passwords do not match.")
            elif len(new_password) < 6:
                st.error("Password must be at least 6 characters long.")
            else:
                success, message = register_user(new_username, new_password, new_name)
                if success:
                    st.success(message)
                    st.session_state['menu_selection'] = 'Login' # Switch to Login view
                    st.rerun()
                else:
                    st.error(message)

def show_login_form():
    """Displays the user login form."""
    st.subheader("Patient Login 🔑")

    with st.form("login_form"):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Login")

        if submitted:
            success, user_name = verify_login(username, password)
            if success:
                st.session_state['logged_in'] = True
                st.session_state['name'] = user_name
                st.session_state['menu_selection'] = 'Login' # Clean up state just before rerun
                st.success(f"Welcome, {user_name}!")
                st.rerun() # Rerun to hit the 'logged_in' branch in the main router
            else:
                st.error("Invalid Username or Password.")

# --- 5. DATA INPUT WIDGETS ---

def user_input_features():
    """Collects all 13 necessary features from the user via the sidebar."""
    st.sidebar.header('Patient Clinical Data')
    st.sidebar.markdown('---')

    age = st.sidebar.slider('Age (years)', 18, 100, 50)
    sex = st.sidebar.selectbox('Sex', options=[1, 0], format_func=lambda x: 'Male (1)' if x == 1 else 'Female (0)')
    cp = st.sidebar.selectbox(
        'Chest Pain Type (cp)',
        options=[0, 1, 2, 3],
        format_func=lambda x: {0: 'Typical Angina', 1: 'Atypical Angina', 2: 'Non-Anginal Pain', 3: 'Asymptomatic'}[x]
    )
    st.sidebar.markdown('---')
    trestbps = st.sidebar.number_input('Resting Blood Pressure (trestbps, mm Hg)', min_value=90, max_value=200, value=120, step=5)
    chol = st.sidebar.number_input('Serum Cholesterol (chol, mg/dl)', min_value=120, max_value=500, value=240, step=5)
    fbs = st.sidebar.selectbox('Fasting Blood Sugar > 120 mg/dl (fbs)', options=[0, 1], format_func=lambda x: 'True (1)' if x == 1 else 'False (0)')
    restecg = st.sidebar.selectbox(
        'Resting ECG Results (restecg)', 
        options=[0, 1, 2],
        format_func=lambda x: {0: 'Normal', 1: 'ST-T Wave Abnormality', 2: 'LV Hypertrophy'}[x]
    )
    st.sidebar.markdown('---')
    thalach = st.sidebar.number_input('Max Heart Rate Achieved (thalach)', min_value=70, max_value=210, value=150, step=5)
    exang = st.sidebar.selectbox('Exercise Induced Angina (exang)', options=[0, 1], format_func=lambda x: 'Yes (1)' if x == 1 else 'No (0)')
    oldpeak = st.sidebar.number_input('ST Depression (oldpeak)', min_value=0.0, max_value=6.5, value=1.0, step=0.1)
    slope = st.sidebar.selectbox(
        'Peak Exercise ST Segment Slope (slope)', 
        options=[0, 1, 2],
        format_func=lambda x: {0: 'Upsloping', 1: 'Flat', 2: 'Downsloping'}[x]
    )
    st.sidebar.markdown('---')
    ca = st.sidebar.slider('Number of Major Vessels Colored (ca)', 0, 3, 0)
    thal = st.sidebar.selectbox(
        'Thalium Stress Test Result (thal)', 
        options=[1, 2, 3],
        format_func=lambda x: {1: 'Normal', 2: 'Fixed Defect', 3: 'Reversible Defect'}[x]
    )

    data = {'age': age, 'sex': sex, 'cp': cp, 'trestbps': trestbps, 'chol': chol, 'fbs': fbs, 'rest