import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import sqlite_utils
import bcrypt
import sqlite3 # Import needed for the thread-safe connection fix
from sqlite_utils.db import NotFoundError

# --- 1. CONFIGURATION AND INITIALIZATION ---

st.set_page_config(
    page_title="Secure Heart Disease Predictor",
    page_icon="❤️",
    layout="wide"
)

# File names (Must match final_setup.py)
MODEL_FILE = 'logistic_model.pkl'
DB_PATH = 'users.db'

# Session State Initialization
if 'logged_in' not in st.session_state:
    st.session_state['logged_in'] = False
if 'menu_selection' not in st.session_state:
    st.session_state['menu_selection'] = 'Login'
if 'name' not in st.session_state:
    st.session_state['name'] = None

# --- 2. MODEL AND DATABASE ACCESS FUNCTIONS ---

@st.cache_resource
def load_model():
    """Load the trained Logistic Regression model."""
    if not os.path.exists(MODEL_FILE):
        st.error(f"FATAL ERROR: Model file '{MODEL_FILE}' not found. Run 'python final_setup.py' first.")
        return None
    try:
        with open(MODEL_FILE, 'rb') as file:
            model = pickle.load(file)
        return model
    except Exception as e:
        st.error(f"FATAL ERROR loading model: {e}")
        return None

@st.cache_resource
def get_db():
    """
    Retrieves the database connection using sqlite3 to ensure check_same_thread=False 
    is applied, resolving threading/stability issues.
    """
    if not os.path.exists(DB_PATH):
        st.error(f"FATAL ERROR: Database file '{DB_PATH}' not found. Run 'python final_setup.py' first.")
        return None
    try:
        # CRITICAL FIX: Use sqlite3.connect with check_same_thread=False
        conn = sqlite3.connect(DB_PATH, check_same_thread=False)
        db = sqlite_utils.Database(conn)
        return db
    except Exception as e:
        st.error(f"FATAL ERROR connecting to database: {e}")
        return None 

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

    data = {'age': age, 'sex': sex, 'cp': cp, 'trestbps': trestbps, 'chol': chol, 'fbs': fbs, 'restecg': restecg, 'thalach': thalach, 'exang': exang, 'oldpeak': oldpeak, 'slope': slope, 'ca': ca, 'thal': thal}
    return pd.DataFrame(data, index=['Input Data'])

# --- 6. MAIN PREDICTION LOGIC ---

def heart_disease_predictor(model):
    """The main body of the heart disease prediction app."""
    st.title(f"Welcome, {st.session_state['name']}! Heart Disease Prediction 🩺")
    
    df_input = user_input_features()
    
    st.subheader('Patient Input Parameters')
    st.dataframe(df_input)

    if st.button('Predict Heart Disease Risk', type='primary'):
        if model is None:
            st.error("Cannot make prediction: Model is not trained or failed to load.")
            return

        input_array = df_input.iloc[0].values.reshape(1, -1)

        try:
            prediction = model.predict(input_array)
            prediction_proba = model.predict_proba(input_array)
            risk_percent = round(prediction_proba[0][1] * 100, 2)

            st.markdown("---")
            st.subheader('Prediction Result')

            if prediction[0] == 1:
                st.error(f"🚨 **HIGH RISK**")
                st.markdown(f"The model predicts a **{risk_percent}%** probability of having Heart Disease.")
                st.warning("Please consult your physician.")
            else:
                st.success(f"✅ **LOW RISK**")
                st.markdown(f"The model predicts a **{risk_percent}%** probability of having Heart Disease.")
                st.info("Maintaining a healthy lifestyle is always recommended.")
            
        except Exception as e:
            st.error(f"Prediction Error: {e}")

# --- 7. APPLICATION ENTRY POINT (FINAL STABLE ROUTER) ---

if __name__ == '__main__':
    # Step 1: Check if setup files exist
    if not os.path.exists(DB_PATH) or not os.path.exists(MODEL_FILE):
        st.error("🛑 CRITICAL SETUP FILES MISSING. Please close the app and run 'python final_setup.py' first.")
        st.stop()

    model = load_model()

    # Step 2: Main Application Router
    
    if st.session_state['logged_in']:
        # If logged_in is TRUE, display the prediction app.
        with st.sidebar:
            st.sidebar.title(f"User: {st.session_state['name']}")
            if st.button("Logout", type='secondary'):
                logout()
        heart_disease_predictor(model)
    else:
        # If logged_in is FALSE, display the access portal (Login/Register).
        st.title("Heart Disease Prediction System ❤️")
        st.markdown("---")
        
        st.sidebar.title("Access Portal")
        # The menu selection widget is placed here
        menu = st.sidebar.radio(
            "Choose Action", 
            ['Login', 'Register'], 
            key='menu_selection_radio',
            index=0 if st.session_state['menu_selection'] == 'Login' else 1
        )
        st.session_state['menu_selection'] = menu

        if st.session_state['menu_selection'] == 'Login':
            show_login_form()
        elif st.session_state['menu_selection'] == 'Register':
            show_registration_form()