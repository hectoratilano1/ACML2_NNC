import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
import joblib
import logging
import os

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

def save_model(model, filename="model.joblib"):
    try:
        current_dir = os.path.dirname(os.path.abspath(__file__))  # /src
        model_path = os.path.abspath(os.path.join(current_dir, "..", filename))  # to project root
        joblib.dump(model, model_path)
        logging.info(f"✅ Model saved to '{model_path}'")
    except Exception as e:
        logging.error(f"❌ Failed to save model: {e}", exc_info=True)

def load_model(filename="model.joblib"):
    try:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.abspath(os.path.join(current_dir, "..", filename))
        logging.info(f"📦 Loading model from: '{model_path}'")
        return joblib.load(model_path)
    except FileNotFoundError:
        logging.error(f"❌ Model file '{filename}' not found at: {model_path}")
        raise
    except Exception as e:
        logging.error(f"❌ Failed to load model: {e}", exc_info=True)
        raise

def predict(model, input_data: pd.DataFrame):
    logging.info("🔮 Making predictions using the loaded model...")
    return model.predict(input_data)
