# -*- coding: utf-8 -*-
"""
Created on Fri Jun 27 18:10:32 2025

@author: palak
"""

import streamlit as st
import pickle

# Load model and feature order
model = pickle.load(open('lung_model (1).sav', "rb"))
feature_order = pickle.load(open('feature_order.pkl', "rb"))

# UI
st.set_page_config(page_title="Lung Cancer Survival Predictor")
st.title("🫁 Lung Cancer Survival Predictor")

col1, col2, col3 = st.columns(3)

with col1:
    age = st.number_input("Age", 1, 120, 60)
    bmi = st.number_input("BMI", 10.0, 60.0, 22.0)
    cholesterol = st.number_input("Cholesterol (mg/dL)", 100, 400, 180)

with col2:
    gender = st.selectbox("Gender", ["Male", "Female"])
    country = st.selectbox("Country", ["USA", "India", "Germany", "UK", "Other"])
    cancer_stage = st.selectbox("Cancer Stage", ["I", "II", "III", "IV"])

with col3:
    treatment_type = st.selectbox("Treatment Type", ["Surgery", "Chemotherapy", "Combined"])
    family_history = st.radio("Family History of Cancer?", ["No", "Yes"])
    smoking_status = st.selectbox("Smoking Status", ["Never", "Passive Smoker", "Former Smoker", "Current Smoker"])
    hypertension = st.radio("Hypertension?", ["No", "Yes"])
    asthma = st.radio("Asthma?", ["No", "Yes"])
    cirrhosis = st.radio("Cirrhosis?", ["No", "Yes"])
    other_cancer = st.radio("Other Cancer?", ["No", "Yes"])

if st.button("Predict Survival"):
    # Encoding manually like in training
    data = {
        'age': age,
        'gender': 1 if gender == "Male" else 0,
        'country': {"USA": 0, "India": 1, "Germany": 2, "UK": 3, "Other": 4}[country],
        'cancer_stage': {"I": 0, "II": 1, "III": 2, "IV": 3}[cancer_stage],
        'family_history': 1 if family_history == "Yes" else 0,
        'smoking_status': {"Never": 0, "Passive Smoker": 1, "Former Smoker": 2, "Current Smoker": 3}[smoking_status],
        'bmi': bmi,
        'cholesterol_level': cholesterol,
        'hypertension': 1 if hypertension == "Yes" else 0,
        'asthma': 1 if asthma == "Yes" else 0,
        'cirrhosis': 1 if cirrhosis == "Yes" else 0,
        'other_cancer': 1 if other_cancer == "Yes" else 0,
        'treatment_type': {"Surgery": 2, "Chemotherapy": 0, "Combined": 1}[treatment_type]
    }

    input_data = [data[feature] for feature in feature_order]

    prediction = model.predict([input_data])[0]

    if prediction == 1:
        st.success("✅ Predicted Outcome: Survived")
    else:
        st.error("❌ Predicted Outcome: Not Survived")
