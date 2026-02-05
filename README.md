# Employee Salary Classification using Machine Learning

This project predicts whether an individual's income is greater than 50K or less than or equal to 50K based on demographic and employment-related attributes. It uses a machine learning classification pipeline with preprocessing, encoding, and model training, deployed through a Streamlit interface.

---

## 🔧 Tech Stack

- Python
- Pandas, NumPy
- Scikit-learn
- Streamlit

---

## 📂 Dataset

- Adult Income dataset containing attributes such as age, education, occupation, work hours, and marital status
- Dataset required preprocessing, cleaning, and encoding before model training

---

## 🧠 Model Used

- Classification model trained on processed dataset
- Feature selection applied to improve input quality

---

## ⚙️ Features

- Data preprocessing and cleaning pipeline
- Encoding categorical features for model compatibility
- Feature selection for relevant inputs
- Streamlit UI for real-time salary prediction
- Structured separation of data, model, and UI logic

---

## 🗂️ Project Structure

Employee-Salray-Prediction/ ├── app.py                # Streamlit interface ├── model.py              # Model training and prediction logic ├── dataset.csv ├── requirements.txt └── README.md

---

## ▶️ How to Run Locally

```bash
git clone https://github.com/ATUL-SHARMA1215/Employee-Salray-Prediction
cd Employee-Salray-Prediction
pip install -r requirements.txt
streamlit run app.py

---

🔍 How It Works

1. Dataset is loaded and cleaned using Pandas
2. Categorical features are encoded for model compatibility
3. Feature selection is applied to improve predictions
4. Classification model is trained on processed data
5. Streamlit interface allows user input for prediction

---

🧪 Testing & Debugging Performed

1. Handled missing and inconsistent data entries
2. Debugged encoding issues in categorical columns
3. Verified prediction outputs across varied inputs
4. Separated preprocessing, model logic, and UI for easier testing

---

📌 Example

Input: Age, education, occupation, work hours
Output: Salary >50K or ≤50K

---

🎯 Learning Outcomes

1. Building classification pipelines on real-world tabular data
2. Handling preprocessing challenges in structured datasets
3. Structuring ML projects with modular Python code
4. Debugging data preprocessing and prediction workflows