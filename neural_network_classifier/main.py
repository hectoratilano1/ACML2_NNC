import logging
from src import preprocessing, model, evaluate

# Set up logging (logs to file and/or console)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    filename="pipeline.log",  # comment this line to log to terminal only
    filemode="w"
)

def run_pipeline():
    try:
        logging.info("Starting neural network pipeline...")

        # Load and preprocess data
        logging.info("Loading and preprocessing data from Admission.csv...")
        df = preprocessing.load_data("data/Admission.csv")
        X, y = preprocessing.prepare_features(df)

        # 🔧 Fix: Strip whitespace from column names
        X.columns = X.columns.str.strip()
        logging.info(f"Cleaned feature names: {X.columns.tolist()}")

        # Split
        logging.info("Splitting data into train and test sets...")
        X_train, X_test, y_train, y_test = model.split_data(X, y)

        # Train
        logging.info("Training the MLP neural network model...")
        clf = model.train_model(X_train, y_train)

        # Save model
        logging.info("Saving trained model to model.joblib...")
        model.save_model(clf)

        # Evaluate
        logging.info("Evaluating model performance...")
        acc, report, matrix = evaluate.evaluate_model(clf, X_test, y_test)

        print(f"\nAccuracy: {acc:.2f}")
        print("\nClassification Report:\n", report)
        print("Confusion Matrix:\n", matrix)

        logging.info(f"Accuracy: {acc:.2f}")
        logging.info("Pipeline finished successfully.")

    except Exception as e:
        logging.error(f"Pipeline execution failed: {e}", exc_info=True)
        print("Something went wrong. Check pipeline.log for details.")

if __name__ == "__main__":
    run_pipeline()
