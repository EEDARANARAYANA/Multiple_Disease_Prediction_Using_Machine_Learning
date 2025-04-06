import streamlit as st
import pandas as pd
import numpy as np
import joblib
from fpdf import FPDF
from datetime import datetime  # Import for date

# Load trained models
def load_model(disease):
    return joblib.load(f"{disease}_model.pkl")

models = {disease: load_model(disease) for disease in ["Diabetes", "Heart_Disease", "Liver_Disease"]}

# Load feature names from a saved file to ensure consistency
feature_columns = joblib.load("feature_columns.pkl")

# Related features for each disease
related_features = {
    "Diabetes": ["Glucose", "BMI", "Insulin", "BloodPressure", "Pregnancies", "SkinThickness"],
    "Heart_Disease": ["chol", "trestbps", "thalach", "fbs", "exang", "oldpeak", "ca"],
    "Liver_Disease": ["Total_Bilirubin", "Direct_Bilirubin", "Alkaline_Phosphotase", "Aspartate_Aminotransferase"]
}

# Streamlit UI with CSS customization
st.markdown("""
    <style>
        body {
            background-color: #f0f2f6;
            font-family: Arial, sans-serif;
        }
        .stButton>button {
            background-color: #4CAF50;
            color: white;
            font-size: 18px;
            padding: 10px 24px;
            border-radius: 8px;
            border: none;
        }
        .stTextInput>div>div>input, .stNumberInput>div>div>input {
            padding: 10px;
            border-radius: 6px;
            border: 1px solid #ccc;
        }
    </style>
""", unsafe_allow_html=True)

st.title("Multi Disease Prediction System")
st.write("Enter patient details to predict the likelihood of diseases.")

# Patient Information
patient_id = st.text_input("Patient ID")
patient_name = st.text_input("Patient Name")

# Input fields
data = []
cols = st.columns(2)
for i, feature in enumerate(feature_columns):
    value = cols[i % 2].number_input(feature, value=0.0)
    data.append(value)

# Function to generate PDF report with Date
def generate_pdf(patient_id, patient_name, predictions, patient_data):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_margins(10, 10, 10)
    pdf.set_font("Arial", "B", 16)
    pdf.cell(0, 10, "Disease Prediction Report", ln=True, align='C')
    pdf.ln(10)

    # Add Current Date
    current_date = datetime.now().strftime("%Y-%m-%d")
    pdf.set_font("Arial", size=12)
    pdf.cell(0, 10, f"Date: {current_date}", ln=True)
    pdf.ln(5)

    pdf.cell(0, 10, f"Patient ID: {patient_id}", ln=True)
    pdf.cell(0, 10, f"Patient Name: {patient_name}", ln=True)
    pdf.ln(10)

    # Prediction Results Table
    pdf.set_font("Arial", "B", 14)
    pdf.cell(0, 10, "Prediction Results:", ln=True)
    pdf.ln(5)
    pdf.set_font("Arial", size=12)
    for disease, result in predictions.items():
        pdf.cell(80, 10, disease, 1)
        pdf.cell(50, 10, "Positive" if result == 1 else "Negative", 1)
        pdf.ln()
    
    pdf.ln(10)
    
    # Feature Values Table (2 Columns)
    pdf.set_font("Arial", "B", 14)
    pdf.cell(0, 10, "Feature Values Provided:", ln=True)
    pdf.ln(5)
    pdf.set_font("Arial", size=12)
    items = list(patient_data.items())
    for i in range(0, len(items), 2):  # 2 columns per row
        row = items[i:i+2]
        for feature, value in row:
            pdf.cell(60, 10, feature, 1)
            pdf.cell(30, 10, str(value), 1)
        pdf.ln()
    
    pdf.ln(10)

    # Related Features
    pdf.set_font("Arial", "B", 14)
    pdf.cell(0, 10, "Related Features for Each Disease:", ln=True)
    pdf.ln(5)
    pdf.set_font("Arial", size=12)
    for disease, features in related_features.items():
        pdf.cell(0, 10, f"{disease}: {', '.join(features)}", ln=True)
    
    pdf_file = "prediction_report.pdf"
    pdf.output(pdf_file)
    return pdf_file

# Predict button
if st.button("Predict"):
    patient_df = pd.DataFrame([data], columns=feature_columns)
    predictions = {disease: models[disease].predict(patient_df)[0] for disease in models}

    st.subheader("Prediction Results")
    for disease, result in predictions.items():
        st.write(f"{disease}: {'Positive' if result == 1 else 'Negative'}")

    patient_data_dict = dict(zip(feature_columns, data))
    pdf_file = generate_pdf(patient_id, patient_name, predictions, patient_data_dict)
    with open(pdf_file, "rb") as file:
        st.download_button(label="Download Report", data=file, file_name=pdf_file, mime="application/pdf")
