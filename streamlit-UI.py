import streamlit as st
from joblib import load
import pandas as pd

# Load trained pipeline
model = load("pipeline.joblib")

# App title
st.title("Student Pass Predictor")

# User inputs
hrs = st.number_input("Study Hours", min_value=0.0, step=0.5)
attendance = st.number_input("Attendance (%)", min_value=0.0, max_value=100.0, step=1.0)

# Prediction button
if st.button("Predict"):
    x = pd.DataFrame({
        "studyHours": [hrs],
        "attendance": [attendance]
    })

    prediction = model.predict(x)

    if prediction[0] == 1:
        st.success("Prediction: PASS")
    else:
        st.error("Prediction: FAIL")
