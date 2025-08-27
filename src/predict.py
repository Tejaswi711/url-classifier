import pandas as pd
import joblib
from .data_preprocessing import clean_url, extract_url_features
from .config import Config


class URLClassifier:
    def __init__(self, model_path, vectorizer_path, pipeline_path):
        self.model = joblib.load(model_path)
        self.vectorizer = joblib.load(vectorizer_path)
        self.pipeline = joblib.load(pipeline_path)

    def predict_single(self, url):
        """Predict class for a single URL."""
        # Preprocess URL
        cleaned_url = clean_url(url)
        url_features = extract_url_features(url)

        # Create feature DataFrame
        feature_df = pd.DataFrame([{
            'cleaned_url': cleaned_url,
            'path_depth': url_features['path_depth'],
            'num_query_params': url_features['num_query_params'],
            'domain': url_features['domain'],
            'has_product_in_path': url_features['has_product_in_path'],
            'has_id_param': url_features['has_id_param'],
            'has_product_pattern': url_features['has_product_pattern'],
            'has_category_pattern': url_features['has_category_pattern']
        }])

        # Prepare features
        X_text = self.vectorizer.transform(feature_df['cleaned_url'])
        X_structured = self.pipeline.transform(feature_df.drop('cleaned_url', axis=1))

        # Combine features and predict
        import scipy.sparse as sp
        X_combined = sp.hstack([X_text, X_structured])

        prediction = self.model.predict(X_combined)
        probability = self.model.predict_proba(X_combined)

        return prediction[0], probability[0]

    def predict_batch(self, urls):
        """Predict classes for a batch of URLs."""
        results = []
        for url in urls:
            try:
                pred, prob = self.predict_single(url)
                results.append({
                    'url': url,
                    'prediction': pred,
                    'confidence': max(prob)
                })
            except Exception as e:
                results.append({
                    'url': url,
                    'prediction': 'error',
                    'confidence': 0.0,
                    'error': str(e)
                })

        return pd.DataFrame(results)


# Example usage
if __name__ == "__main__":
    classifier = URLClassifier(
        model_path="../models/trained_model.pkl",
        vectorizer_path="../models/tfidf_vectorizer.pkl",
        pipeline_path="../models/feature_pipeline.pkl"
    )

    # Example URLs to classify
    test_urls = [
        "https://www.amazon.in/dp/B09G99CW2T",
        "https://www.amazon.in/s?k=laptops",
        "https://www.amazon.in/gp/help/customer/display.html"
    ]

    results = classifier.predict_batch(test_urls)
    print(results)