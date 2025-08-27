from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib


def create_text_features(df, text_column='cleaned_url', max_features=2000, ngram_range=(1, 2)):
    """Create TF-IDF features from text."""
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        stop_words='english',
        min_df=2,
        max_df=0.95
    )

    X_text = vectorizer.fit_transform(df[text_column])
    return X_text, vectorizer


def create_feature_pipeline():
    """Create feature engineering pipeline."""
    numeric_features = ['path_depth', 'num_query_params']
    categorical_features = ['domain', 'has_product_in_path', 'has_id_param',
                            'has_product_pattern', 'has_category_pattern']

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ])

    return preprocessor


def prepare_features(df, text_vectorizer=None, feature_pipeline=None, fit=True):
    """Prepare final feature set."""
    # Text features
    if fit:
        X_text, text_vectorizer = create_text_features(df)
    else:
        X_text = text_vectorizer.transform(df['cleaned_url'])

    # Structured features
    structured_features = df[['path_depth', 'num_query_params', 'domain',
                              'has_product_in_path', 'has_id_param',
                              'has_product_pattern', 'has_category_pattern']]

    if fit:
        X_structured = feature_pipeline.fit_transform(structured_features)
    else:
        X_structured = feature_pipeline.transform(structured_features)

    # Combine features
    import scipy.sparse as sp
    X_combined = sp.hstack([X_text, X_structured])

    if fit:
        return X_combined, text_vectorizer, feature_pipeline
    else:
        return X_combined