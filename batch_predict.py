import pandas as pd
from src.predict import URLClassifier


def batch_predict_urls(csv_file_path, output_file):
    """Predict classes for URLs in a CSV file."""

    # Load URLs from CSV
    df = pd.read_csv(csv_file_path)

    # Initialize classifier
    classifier = URLClassifier(
        model_path="models/trained_model.pkl",
        vectorizer_path="models/tfidf_vectorizer.pkl",
        pipeline_path="models/feature_pipeline.pkl"
    )

    # Make predictions
    results = classifier.predict_batch(df['url'].tolist())

    # Save results
    results.to_csv(output_file, index=False)
    print(f"Predictions saved to: {output_file}")
    print(f"Results summary:")
    print(results['prediction'].value_counts())

    return results


# Example usage
if __name__ == "__main__":
    # Create a sample CSV with URLs to classify
    sample_urls = [
        "https://www.amazon.in/dp/B09G99CW2T",
        "https://www.amazon.in/s?k=laptops",
        "https://www.flipkart.com/search?q=smartphones",
        "https://www.flipkart.com/product/p/12345"
    ]

    sample_df = pd.DataFrame({'url': sample_urls})
    sample_df.to_csv('data/sample_urls_to_classify.csv', index=False)

    # Run batch prediction
    results = batch_predict_urls('data/sample_urls_to_classify.csv', 'data/predictions_output.csv')
    print(results)