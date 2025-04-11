import streamlit as st
import pandas as pd
import os
from src.model import load_model, predict

st.title("🎓 Admission Prediction (Neural Network)")
st.write("Enter student details to estimate chance of admission:")

# Input fields
gre = st.number_input("GRE Score", min_value=260, max_value=340)
toefl = st.number_input("TOEFL Score", min_value=0)
univ_rating = st.selectbox("University Rating", [1, 2, 3, 4, 5])
sop = st.slider("SOP Strength (1-5)", 1.0, 5.0, step=0.5)
lor = st.slider("LOR Strength (1-5)", 1.0, 5.0, step=0.5)
cgpa = st.number_input("CGPA (0-10)", min_value=0.0, max_value=10.0, step=0.1)
research = st.selectbox("Research Experience", ["No", "Yes"])

# Try loading the model
try:
    model = load_model()
except Exception as e:
    st.error(f"❌ Could not load model: {e}")
    st.stop()

# Build input dictionary
input_dict = {
    "GRE_Score": gre,
    "TOEFL_Score": toefl,
    "SOP": sop,
    "LOR": lor,
    "CGPA": cgpa,
    f"University_Rating_{univ_rating}": 1,
    "Research_1" if research == "Yes" else "Research_0": 1
}

# Ensure all expected model inputs are covered
expected_cols = list(model.feature_names_in_)
for col in expected_cols:
    input_dict.setdefault(col, 0)

# Format input as DataFrame
input_df = pd.DataFrame([input_dict])[expected_cols]

# Prediction
if st.button("Predict Admission Chance"):
    try:
        result = predict(model, input_df)[0]
        if result == 1:
            st.success("✅ High chance of admission!")
        else:
            st.error("❌ Low chance of admission.")
    except Exception as e:
        st.error(f"Prediction error: {e}")
