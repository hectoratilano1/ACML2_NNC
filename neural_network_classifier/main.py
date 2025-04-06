
from src import preprocessing, model, evaluate

def run_pipeline():
    # Load and preprocess data
    df = preprocessing.load_data("data/Admission.csv")
    X, y = preprocessing.prepare_features(df)

    # Split
    X_train, X_test, y_train, y_test = model.split_data(X, y)

    # Train model
    clf = model.train_model(X_train, y_train)

    # Save model
    model.save_model(clf)

    # Evaluate
    acc, report, matrix = evaluate.evaluate_model(clf, X_test, y_test)
    print(f"Accuracy: {acc:.2f}")
    print("\nClassification Report:\n", report)
    print("Confusion Matrix:\n", matrix)

if __name__ == "__main__":
    run_pipeline()
