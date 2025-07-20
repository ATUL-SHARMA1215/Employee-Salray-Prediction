import streamlit as st
import pandas as pd
import joblib

# Load model and encoders
model = joblib.load("model/best_model.pkl")
scaler = joblib.load("model/scaler.pkl")
label_encoders = joblib.load("model/label_encoders.pkl")

st.set_page_config(page_title="💼 Salary Predictor", layout="centered")

# Background and Styling
st.markdown("""
    <style>
    .stApp {
        background-image: url("https://images.unsplash.com/photo-1504384308090-c894fdcc538d?ixlib=rb-4.0.3&auto=format&fit=crop&w=1920&q=80");
        background-size: cover;
        background-attachment: fixed;
        background-position: center;
        color: white;
    }
    .main-container {
        background-color: rgba(0, 0, 0, 0.6);
        padding: 2rem;
        border-radius: 20px;
        max-width: 720px;
        margin: 0 auto;
        box-shadow: 0 0 15px rgba(0,0,0,0.4);
    }
    h1 {
        text-align: center;
        color: #ffffff;
        font-size: 3rem;
        text-shadow: 2px 2px 4px #000;
        margin-bottom: 2rem;
    }
    .result {
        background: rgba(255, 255, 255, 0.9);
        color: black;
        padding: 20px;
        border-radius: 15px;
        margin-top: 20px;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-container">', unsafe_allow_html=True)
st.markdown("<h1>💼 Salary Predictor</h1>", unsafe_allow_html=True)

# UI Form
with st.form("predict_form"):
    age = st.slider("Age", 18, 90, 30)
    education = st.selectbox("Education", label_encoders["education"].classes_)
    occupation = st.selectbox("Occupation", label_encoders["occupation"].classes_)
    capital_gain = st.number_input("Capital Gain", 0, 99999, value=0)
    relationship = st.selectbox("Relationship", label_encoders["relationship"].classes_)
    hours_per_week = st.slider("Hours/Week", 1, 99, 40)
    fnlwgt = st.number_input("FNLWGT", 10000, 1000000, value=100000, step=1000)
    educational_num = st.slider("Educational-Num", 1, 16, 10)
    race = st.selectbox("Race", label_encoders["race"].classes_)
    capital_loss = st.number_input("Capital Loss", 0, 99999, value=0)
    marital_status = st.selectbox("Marital Status", label_encoders["marital_status"].classes_)
    gender = st.selectbox("Gender", label_encoders["gender"].classes_)
    native_country = st.selectbox("Native Country", label_encoders["native_country"].classes_)
    workclass = st.selectbox("Workclass", label_encoders["workclass"].classes_)

    submit = st.form_submit_button("🔍 Predict")

st.markdown('</div>', unsafe_allow_html=True)

# Prediction Logic
if submit:
    input_data = {
        'age': age,
        'workclass': workclass,
        'fnlwgt': fnlwgt,
        'education': education,
        'educational_num': educational_num,
        'marital_status': marital_status,
        'occupation': occupation,
        'relationship': relationship,
        'race': race,
        'gender': gender,
        'capital_gain': capital_gain,
        'capital_loss': capital_loss,
        'hours_per_week': hours_per_week,
        'native_country': native_country
    }

    df = pd.DataFrame([input_data])
    for col in df.select_dtypes(include='object').columns:
        df[col] = label_encoders[col].transform(df[col])

    df = df[list(input_data.keys())]
    scaled = scaler.transform(df)
    prediction = model.predict(scaled)[0]
    income_label = label_encoders["income"].inverse_transform([prediction])[0]
    prob = model.predict_proba(scaled).max()

    # Output Result
    st.markdown(f"""
        <div class="result">
            <h3>🎯 Prediction Result</h3>
            <p><strong>💰 Income Category:</strong> {income_label}</p>
            <p><strong>📈 Confidence:</strong> {prob*100:.2f}%</p>
        </div>
    """, unsafe_allow_html=True)
