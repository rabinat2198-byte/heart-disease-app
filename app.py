import streamlit as st
import joblib
import pandas as pd

st.set_page_config(page_title="Heart Disease Predictor", page_icon="❤️")

@st.cache_resource
def load_model():
    return joblib.load("heart_model.pkl")

model = load_model()

FEATURES = ["age","sex","cp","trestbps","chol","fbs","restecg",
            "thalach","exang","oldpeak","slope","ca","thal"]

st.title("❤️ Heart Disease Predictor")
st.caption("Enter the patient details and click Predict. Educational project only, not medical advice.")

col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", 1, 120, 55)
    sex = st.selectbox("Sex", [1, 0], format_func=lambda x: "Male" if x == 1 else "Female")
    cp = st.selectbox("Chest pain type", [0, 1, 2, 3], format_func=lambda x: {
        0: "0 - Typical angina", 1: "1 - Atypical angina",
        2: "2 - Non-anginal pain", 3: "3 - Asymptomatic"}[x])
    trestbps = st.number_input("Resting blood pressure (mm Hg)", 80, 220, 130)
    chol = st.number_input("Cholesterol (mg/dl)", 100, 600, 240)
    fbs = st.selectbox("Fasting blood sugar > 120 mg/dl", [0, 1],
                       format_func=lambda x: "Yes" if x == 1 else "No")
    restecg = st.selectbox("Resting ECG result", [0, 1, 2], format_func=lambda x: {
        0: "0 - Normal", 1: "1 - ST-T abnormality", 2: "2 - LV hypertrophy"}[x])

with col2:
    thalach = st.number_input("Max heart rate achieved", 60, 220, 150)
    exang = st.selectbox("Exercise induced angina", [0, 1],
                         format_func=lambda x: "Yes" if x == 1 else "No")
    oldpeak = st.number_input("ST depression (oldpeak)", 0.0, 7.0, 1.0, step=0.1)
    slope = st.selectbox("Slope of peak exercise ST", [0, 1, 2], format_func=lambda x: {
        0: "0 - Upsloping", 1: "1 - Flat", 2: "2 - Downsloping"}[x])
    ca = st.number_input("Major vessels colored (0-4)", 0, 4, 0)
    thal = st.selectbox("Thalassemia", [0, 1, 2, 3], format_func=lambda x: {
        0: "0", 1: "1 - Fixed defect", 2: "2 - Normal", 3: "3 - Reversible defect"}[x])

if st.button("Predict", type="primary", use_container_width=True):
    values = [age, sex, cp, trestbps, chol, fbs, restecg,
              thalach, exang, oldpeak, slope, ca, thal]
    row = pd.DataFrame([values], columns=FEATURES)
    pred = int(model.predict(row)[0])
    prob = float(model.predict_proba(row)[0][1])

    if pred == 1:
        st.error("⚠️ High risk of heart disease")
    else:
        st.success("✅ Low risk of heart disease")
    st.metric("Probability of disease", f"{prob*100:.1f}%")
    st.progress(prob)
