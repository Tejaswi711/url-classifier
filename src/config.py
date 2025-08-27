import os


class Config:
    # Use absolute paths
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # Data paths
    DATA_PATH = os.path.join(BASE_DIR, "data", "labeled_samples.csv")
    MODEL_SAVE_PATH = os.path.join(BASE_DIR, "models", "trained_model.pkl")
    VECTORIZER_SAVE_PATH = os.path.join(BASE_DIR, "models", "tfidf_vectorizer.pkl")

    # Preprocessing
    TEST_SIZE = 0.2
    RANDOM_STATE = 42

    # Feature Engineering
    TFIDF_MAX_FEATURES = 2000
    NGRAM_RANGE = (1, 2)

    # Model parameters
    MODEL_TYPE = "random_forest"  # Options: "random_forest", "logistic_regression", "svm"

    # Label mapping
    LABEL_MAPPING = {
        'product': 'product',
        'category': 'category',
        'other': 'other',
        'not_product': 'other'
    }