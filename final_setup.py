# In final_setup.py

import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Add this import for balancing the data
from imblearn.over_sampling import SMOTE 
# Add this for pipeline setup, though not strictly required, it's good practice
from imblearn.pipeline import Pipeline 

def train_and_save_model():
    """Trains the robust Logistic Regression model using SMOTE and saves it."""
    print("\n--- 2. Training Robust Model ---")
    
    # ... (File check and DataFrame loading code remains the same) ...
    try:
        df = pd.read_csv(DATA_FILE)
    except Exception as e:
        print(f"❌ FATAL ERROR reading {DATA_FILE}: {e}")
        return False
        
    FEATURES = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 
                'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']
    TARGET = 'target'
    
    X = df[FEATURES].fillna(df[FEATURES].mean())
    y = df[TARGET]
    
    # Split data before balancing to prevent data leakage
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 1. Apply SMOTE to the training data to balance the classes
    print("   Applying SMOTE to balance the training data...")
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

    # 2. Configure a regularized and stable Logistic Regression model
    logreg = LogisticRegression(
        max_iter=5000,          # Increased iterations for convergence
        solver='liblinear',     # Good for smaller datasets
        penalty='l2',           # Use L2 regularization to prevent extreme coefficients
        C=0.5,                  # Regularization strength (smaller C = stronger regularization)
        random_state=42
    )

    logreg.fit(X_train_resampled, y_train_resampled)
    
    # Evaluate model accuracy on the test set
    test_accuracy = accuracy_score(y_test, logreg.predict(X_test))
    
    # Save the trained model
    with open(MODEL_FILE, 'wb') as file:
        pickle.dump(logreg, file)
        
    print(f"✅ Model trained and saved as '{MODEL_FILE}'.")
    print(f"   Accuracy on Test Set (Aiming for 80%+): {test_accuracy*100:.2f}%")
    return True