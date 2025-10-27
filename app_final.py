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

# File names
MODEL_FILE = 'logistic_model.pkl'
DB_PATH = 'users.db'
DATA_FILE = 'heart.csv' 

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
        st.error(f"FATAL DB ERROR: Could not ensure 'patients' table exists: {e}")
        return False
    return True

@st.cache_resource
def get_db():
    """Retrieves the database connection."""
    try:
        conn = sqlite3.connect(DB_PATH, check_same_thread=False)
        db = sqlite_utils.Database(conn)
        
        if not initialize_database(db):
            return None
            
        return db
    except Exception as e:
        st.error(f"FATAL ERROR connecting to database: {e}")
        return None 


@st.cache_resource(show_spinner="Training model on first run...")
def run_setup():
    """Consolidated function to train the model and save it locally."""
    if os.path.exists(MODEL_FILE):
        try:
            with open(MODEL_FILE, 'rb') as file:
                model = pickle.load(file)
            return model
        except Exception:
            st.warning("Existing model file corrupted. Retraining model now...")
            if os.path.exists(MODEL_FILE):
                os.remove(MODEL_FILE) 

    # --- MODEL TRAINING LOGIC ---
    st.info("Model file not found. Training robust model now (this may take a moment)...")
    
    if not os.path.exists(DATA_FILE):
        st.error(f"❌ FATAL ERROR: Data file '{DATA_FILE}' not found. Ensure it is in the project directory.")
        st.stop()
        
    try:
        df = pd.read_csv(DATA_FILE)
    except Exception as e:
        st.error(f"❌ FATAL ERROR reading {DATA_FILE}: {e}")
        st.stop()
    
    # 1. DATA COERCION AND CLEANUP
    st.info(f"Columns found in {DATA_FILE}: {list(df.columns)}")
    for col in df.columns:
        # Coerce all columns to numeric. Errors (like '?' or other non-numerics) become NaN.
        df[col] = pd.to_numeric(df[col], errors='coerce')
        
    # Standard features identified from your data structure
    FEATURES = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 
                'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal'] 
    TARGET = 'num' 
    
    # -------------------------------------------------------------
    # CRITICAL FIXES FOR NaN VALUES
    # -------------------------------------------------------------
    
    # a. Drop rows where the TARGET variable ('num') is NaN.
    df = df.dropna(subset=[TARGET])
    
    # b. Binarize the target: Convert 0, 1, 2, 3, 4 to 0 (No Disease) or 1 (Disease).
    y = df[TARGET].apply(lambda x: 1 if x > 0 else 0)
    
    # c. Create Feature Matrix X
    X = df[FEATURES]
    
    # d. CRITICAL FIX: Drop columns that are all NaN after dropping rows. 
    # This prevents fillna(X.mean()) from failing.
    X = X.dropna(axis=1, how='all')
    
    # e. Reset FEATURES list to reflect any columns dropped above (like 'ca' or 'thal')
    actual_features = list(X.columns)
    
    # f. Impute remaining missing values in FEATURES using the mean.
    X = X.fillna(X.mean())
    
    # -------------------------------------------------------------
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # --- DIAGNOSTIC: Check training sample size ---
    st.info(f"Training samples (total): {len(X_train)}")
    st.info(f"Training samples (target=1, minority): {y_train.sum()}")
    # --- END DIAGNOSTIC ---

    # Define the Robust Pipeline
    steps = [
        ('scaler', StandardScaler()),                    
        ('smote', SMOTE(random_state=42, k_neighbors=2)), 
        ('logreg', LogisticRegression(
            max_iter=5000, 
            solver='liblinear', 
            penalty='l2', 
            C=0.5, 
            random_state=42
        ))                                           
    ]
    pipeline = Pipeline(steps=steps)

    # Train the Pipeline
    pipeline.fit(X_train, y_train)
    
    # Save the trained pipeline model
    with open(MODEL_FILE, 'wb') as file:
        # Save the model AND the final feature list to ensure prediction works correctly
        pickle.dump((pipeline, actual_features), file)
    
    st.success("Model training complete!")
    return (pipeline, actual_features)


# --- 3. USER AUTHENTICATION FUNCTIONS (UNCHANGED) ---

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

# --- 4. LOGIN/REGISTRATION FORM DISPLAY (UNCHANGED) ---

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
                    st.session_state['menu_selection'] = 'Login' 
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
                st.session_state['menu_selection'] = 'Login' 
                st.success(f"Welcome, {user_name}!")
                st.rerun() 
            else:
                st.error("Invalid Username or Password.")

# --- 5. DATA INPUT WIDGETS (UPDATED to accept dynamic feature list) ---

def user_input_features(feature_names):
    """Collects features based on the list of features the model was trained on."""
    st.sidebar.header('Patient Clinical Data')
    st.sidebar.markdown('---')
    
    # Store all inputs temporarily
    input_data = {}

    # Define a consistent order and retrieve input using the correct names
    # Only create widgets for features the model was trained on
    
    # Define mapping for all possible features
    feature_widgets = {
        'age': st.sidebar.slider('Age (years)', 18, 100, 50),
        'sex': st.sidebar.selectbox('Sex', options=[1, 0], format_func=lambda x: 'Male (1)' if x == 1 else 'Female (0)'),
        'cp': st.sidebar.selectbox(
            'Chest Pain Type (cp)',
            options=[0, 1, 2, 3],
            format_func=lambda x: {0: 'Typical Angina', 1: 'Atypical Angina', 2: 'Non-Anginal Pain', 3: 'Asymptomatic'}[x]
        ),
        'trestbps': st.sidebar.number_input('Resting Blood Pressure (trestbps, mm Hg)', min_value=90, max_value=200, value=120, step=5),
        'chol': st.sidebar.number_input('Serum Cholesterol (chol, mg/dl)', min_value=120, max_value=500, value=240, step=5),
        'fbs': st.sidebar.selectbox('Fasting Blood Sugar > 120 mg/dl (fbs)', options=[0, 1], format_func=lambda x: 'True (1)' if x == 1 else 'False (0)'),
        'restecg': st.sidebar.selectbox(
            'Resting ECG Results (restecg)', 
            options=[0, 1, 2],
            format_func=lambda x: {0: 'Normal', 1: 'ST-T Wave Abnormality', 2: 'LV Hypertrophy'}[x]
        ),
        'thalach': st.sidebar.number_input('Max Heart Rate Achieved (thalach)', min_value=70, max_value=210, value=150, step=5),
        'exang': st.sidebar.selectbox('Exercise Induced Angina (exang)', options=[0, 1], format_func=lambda x: 'Yes (1)' if x == 1 else 'No (0)'),
        'oldpeak': st.sidebar.number_input('ST Depression (oldpeak)', min_value=0.0, max_value=6.5, value=1.0, step=0.1),
        'slope': st.sidebar.selectbox(
            'Peak Exercise ST Segment Slope (slope)', 
            options=[0, 1, 2],
            format_func=lambda x: {0: 'Upsloping', 1: 'Flat', 2: 'Downsloping'}[x]
        ),
        'ca': st.sidebar.slider('Number of Major Vessels Colored (ca)', 0, 3, 0),
        'thal': st.sidebar.selectbox(
            'Thalium Stress Test Result (thal)', 
            options=[1, 2, 3],
            format_func=lambda x: {1: 'Normal', 2: 'Fixed Defect', 3: 'Reversible Defect'}[x]
        )
    }

    # Only include features the model was trained on
    for feature in feature_names:
        if feature in feature_widgets:
            input_data[feature] = feature_widgets[feature]
        # Note: If a feature used for training isn't in this dictionary, it will be skipped.
        
    return pd.DataFrame(input_data, index=['Input Data'])[[f for f in feature_names if f in input_data]] # Ensure correct column order

# --- 6. MAIN PREDICTION LOGIC ---

def heart_disease_predictor(model_and_features):
    """The main body of the heart disease prediction app."""
    st.title(f"Welcome, {st.session_state['name']}! Heart Disease Prediction 🩺")
    
    pipeline, actual_features = model_and_features
    
    df_input = user_input_features(actual_features)
    
    st.subheader('Patient Input Parameters')
    st.dataframe(df_input)

    if st.button('Predict Heart Disease Risk', type='primary'):
        if pipeline is None:
            st.error("Cannot make prediction: Model is not trained or failed to load.")
            return

        # Ensure input data columns match the training order/subset
        input_array = df_input.iloc[0].values.reshape(1, -1)

        try:
            prediction_proba = pipeline.predict_proba(input_array)
            risk_percent = round(prediction_proba[0][1] * 100, 2)
            
            CLASSIFICATION_THRESHOLD = 0.30 

            if prediction_proba[0][1] > CLASSIFICATION_THRESHOLD:
                prediction_label = 1 # High Risk
            else:
                prediction_label = 0 # Low Risk

            st.markdown("---")
            st.subheader('Prediction Result')

            if prediction_label == 1:
                st.error(f"🚨 **HIGH RISK**")
                st.markdown(f"The model predicts a **{risk_percent}%** probability of having Heart Disease (using a conservative threshold of {CLASSIFICATION_THRESHOLD}).")
                st.warning("Please consult your physician.")
            else:
                st.success(f"✅ **LOW RISK**")
                st.markdown(f"The model predicts a **{risk_percent}%** probability of having Heart Disease (using a conservative threshold of {CLASSIFICATION_THRESHOLD}).")
                st.info("Maintaining a healthy lifestyle is always recommended.")
            
        except Exception as e:
            st.error(f"Prediction Error: {e}")

# --- 7. APPLICATION ENTRY POINT (FINAL STABLE ROUTER) ---

if __name__ == '__main__':
    
    # 1. Run Setup/Load Model
    # run_setup returns (pipeline, actual_features_list)
    model_and_features = run_setup()
    
    if model_and_features is None:
        st.error("🛑 Cannot run application without a trained model.")
        st.stop()

    # 2. Get Database Connection
    db = get_db()
    if db is None:
        st.stop()
        
    # 3. Application Router
    
    if st.session_state['logged_in']:
        # Logged In View
        with st.sidebar:
            st.sidebar.title(f"User: {st.session_state['name']}")
            if st.button("Logout", type='secondary'):
                logout()
        heart_disease_predictor(model_and_features)
    else:
        # Login/Registration View
        st.title("Heart Disease Prediction System ❤️")
        st.markdown("---")
        
        st.sidebar.title("Access Portal")
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