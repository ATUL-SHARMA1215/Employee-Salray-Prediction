# train_model.py

import pandas as pd
import numpy as np
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

DATA_PATH = "data/adult 3.csv"

def load_data(path):
    try:
        df = pd.read_csv(path)
        print(f"✅ Loaded dataset with shape: {df.shape}")
        return df
    except Exception as e:
        raise ValueError(f"❌ Failed to load CSV: {e}")

def preprocess_data(df):
    # Drop rows with missing or malformed data
    df.replace("?", np.nan, inplace=True)
    df.dropna(inplace=True)

    # Standardize column names
    df.columns = df.columns.str.strip().str.lower().str.replace('-', '_')

    # Encode categorical features
    cat_cols = df.select_dtypes(include='object').columns
    label_encoders = {}
    for col in cat_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le

    return df, label_encoders

def train_and_save_model(df, label_encoders):
    # Split features/target
    target_col = 'income'
    if target_col not in df.columns:
        raise ValueError(f"❌ Target column '{target_col}' not found in dataset!")

    X = df.drop(target_col, axis=1)
    y = df[target_col]

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Scale numeric features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train_scaled, y_train)

    # Evaluation
    y_pred = model.predict(X_test_scaled)
    print("🎯 Accuracy:", accuracy_score(y_test, y_pred))
    print("📊 Classification Report:\n", classification_report(y_test, y_pred))

    # Save model and tools
    os.makedirs("model", exist_ok=True)
    joblib.dump(model, "model/best_model.pkl")
    joblib.dump(scaler, "model/scaler.pkl")
    joblib.dump(label_encoders, "model/label_encoders.pkl")
    print("✅ Model and preprocessors saved!")

if __name__ == "__main__":
    df = load_data(DATA_PATH)
    df, label_encoders = preprocess_data(df)
    train_and_save_model(df, label_encoders)
