
import pandas as pd

def load_data(filepath: str) -> pd.DataFrame:
    df = pd.read_csv(filepath)

    # Convert target variable to binary
    df['Admit_Chance'] = (df['Admit_Chance'] >= 0.8).astype(int)

    # Drop Serial_No
    if 'Serial_No' in df.columns:
        df = df.drop(columns=['Serial_No'])

    # Convert some columns to categorical (if needed)
    df['University_Rating'] = df['University_Rating'].astype('object')
    df['Research'] = df['Research'].astype('object')

    return df

def prepare_features(df: pd.DataFrame):
    X = df.drop(columns=['Admit_Chance'])
    X = pd.get_dummies(X)  # one-hot encode categorical
    y = df['Admit_Chance']
    return X, y
