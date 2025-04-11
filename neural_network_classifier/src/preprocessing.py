import pandas as pd
import streamlit as st

def load_data(filepath: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(filepath)
    except FileNotFoundError:
        st.error(f"❌ File not found: {filepath}")
        st.stop()

    # Convert target variable to binary
    df['Admit_Chance'] = (df['Admit_Chance'] >= 0.8).astype(int)

    # Drop Serial_No if it exists
    if 'Serial_No' in df.columns:
        df = df.drop(columns=['Serial_No'])

    # Convert specified columns to categorical
    df['University_Rating'] = df['University_Rating'].astype('object')
    df['Research'] = df['Research'].astype('object')

    return df

def prepare_features(df: pd.DataFrame):
    X = df.drop(columns=['Admit_Chance'])
    X = pd.get_dummies(X)  # One-hot encode categorical variables
    y = df['Admit_Chance']
    return X, y
