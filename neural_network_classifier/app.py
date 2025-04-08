import streamlit as st
import pandas as pd
from src.model import load_model, predict

st.title("🎓 Admission Prediction (Neural Network)")

st.write("Enter student details to estimate chance of admission:")

gre = st.number_input("GRE Score", min_value=260, max_value=340)
toefl = st.number_input("TOEFL Score", min_value=0)
univ_rating = st.selectbox("University Rating", [1, 2, 3, 4, 5])
sop = st.slider("SOP Strength (1-5)", 1.0, 5.0, step=0.5)
lor = st.slider("LOR Strength (1-5)", 1.0, 5.0, step=0.5)
cgpa = st.number_input("CGPA (0-10)", min_value=0.0, max_value=10.0, step=0.1)
research = st.selectbox("Research Experience", ["No", "Yes"])

# Load model once
model = load_model()

# Build input dict (initial values)
input_dict = {
    "GRE_Score": gre,
    "TOEFL_Score": toefl,
    "SOP": sop,
    "LOR": lor,
    "CGPA": cgpa,
    f"University_Rating_{univ_rating}": 1,
    "Research_1" if research == "Yes" else "Research_0": 1
}

# Let the model tell us which features it was trained on
expected_cols = list(model.feature_names_in_)

# Fill missing dummy variables with 0
for col in expected_cols:
    input_dict.setdefault(col, 0)

# Create final input DataFrame in correct order
input_df = pd.DataFrame([input_dict])[expected_cols]

# Prediction
if st.button("Predict Admission Chance"):
    result = predict(model, input_df)[0]
    if result == 1:
        st.success("✅ High chance of admission!")
    else:
        st.error("❌ Low chance of admission.")
