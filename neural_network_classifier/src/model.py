import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
import joblib
import logging

# Optional fallback config (safe if used standalone)
logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")

def split_data(X, y):
    logging.info("Splitting dataset into training and test sets...")
    return train_test_split(X, y, test_size=0.2, random_state=42)

def train_model(X_train, y_train):
    logging.info("Training neural network (MLPClassifier)...")
    clf = MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42)
    clf.fit(X_train, y_train)
    logging.info("Neural network training complete.")
    return clf

def save_model(model, path="model.joblib"):
    try:
        joblib.dump(model, path)
        logging.info(f"Model saved to '{path}'")
    except Exception as e:
        logging.error(f"Failed to save model: {e}", exc_info=True)

def load_model(path="model.joblib"):
    try:
        logging.info(f"Loading model from '{path}'...")
        return joblib.load(path)
    except FileNotFoundError:
        logging.error(f"Model file '{path}' not found.")
        raise
    except Exception as e:
        logging.error(f"Failed to load model: {e}", exc_info=True)
        raise

def predict(model, input_data: pd.DataFrame):
    logging.info("Making predictions using the loaded model...")
    return model.predict(input_data)
