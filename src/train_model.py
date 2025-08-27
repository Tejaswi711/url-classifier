import pandas as pd
import joblib
import os  # Add this import
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score

from data_preprocessing import load_data, preprocess_data, split_data
from feature_engineering import prepare_features, create_feature_pipeline
from config import Config


def train_model():
    """Main training function."""
    config = Config()

    # Create directories if they don't exist
    os.makedirs(os.path.dirname(config.MODEL_SAVE_PATH), exist_ok=True)
    os.makedirs(os.path.dirname(config.VECTORIZER_SAVE_PATH), exist_ok=True)
    os.makedirs("../evaluation", exist_ok=True)

    print("Loading and preprocessing data...")
    df = load_data(config.DATA_PATH)
    df_processed = preprocess_data(df, config.LABEL_MAPPING)

    print("Splitting data...")
    train_df, test_df = split_data(df_processed, config.TEST_SIZE, config.RANDOM_STATE)

    print("Engineering features...")
    feature_pipeline = create_feature_pipeline()
    X_train, text_vectorizer, feature_pipeline = prepare_features(
        train_df, feature_pipeline=feature_pipeline, fit=True
    )
    y_train = train_df['label']

    X_test = prepare_features(
        test_df, text_vectorizer, feature_pipeline, fit=False
    )
    y_test = test_df['label']

    print("Training model...")
    if config.MODEL_TYPE == "random_forest":
        model = RandomForestClassifier(
            n_estimators=100,
            random_state=config.RANDOM_STATE,
            class_weight='balanced'
        )
    elif config.MODEL_TYPE == "logistic_regression":
        model = LogisticRegression(
            random_state=config.RANDOM_STATE,
            class_weight='balanced',
            max_iter=1000
        )
    elif config.MODEL_TYPE == "svm":
        model = SVC(
            random_state=config.RANDOM_STATE,
            class_weight='balanced',
            probability=True
        )

    model.fit(X_train, y_train)

    print("Evaluating model...")
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    print(f"Test Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    # Save model and artifacts
    print("Saving model...")
    joblib.dump(model, config.MODEL_SAVE_PATH)
    joblib.dump(text_vectorizer, config.VECTORIZER_SAVE_PATH)
    joblib.dump(feature_pipeline, "../models/feature_pipeline.pkl")

    # Save evaluation report
    report = classification_report(y_test, y_pred, output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    report_df.to_csv("../evaluation/classification_report.csv")

    print("Training completed successfully!")
    print(f"Model saved to: {config.MODEL_SAVE_PATH}")
    print(f"Vectorizer saved to: {config.VECTORIZER_SAVE_PATH}")


if __name__ == "__main__":
    train_model()