import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.predict import URLClassifier


def test_predictions():
    # Initialize classifier
    classifier = URLClassifier(
        model_path="models/trained_model.pkl",
        vectorizer_path="models/tfidf_vectorizer.pkl",
        pipeline_path="models/feature_pipeline.pkl"
    )

    # Test URLs from your dataset
    test_urls = [
        "https://www.amazon.in/dp/B09G99CW2T",  # Product
        "https://www.amazon.in/s?k=noise+crew+smartwatch+1.38",  # Not product
        "https://www.flipkart.com/apple-iphone-13-blue-128-gb/p/itm6e3b5b5a7cfd9",  # Product
        "https://www.flipkart.com/search?q=noise+watch",  # Not product
        "https://www.google.com/shopping/product/16848581607082948437",  # Product
        "https://www.google.com/search?q=iphone+13",  # Not product
    ]

    print("Testing URL Classification Model")
    print("=" * 60)

    for url in test_urls:
        try:
            prediction, confidence = classifier.predict_single(url)
            confidence_score = max(confidence) * 100
            print(f"URL: {url[:50]}...")
            print(f"   Prediction: {prediction}")
            print(f"   Confidence: {confidence_score:.1f}%")
            print(
                f"   Expected: {'product' if 'dp/' in url or '/p/' in url else 'other'}")
            print("-" * 40)
        except Exception as e:
            print(f"Error with URL: {url}")
            print(f"   Error: {e}")
            print("-" * 40)


if __name__ == "__main__":
    test_predictions()