import pandas as pd
import re
from urllib.parse import urlparse, parse_qs
from sklearn.model_selection import train_test_split
from src.config import Config


def load_data(file_path):
    """Load and preprocess the dataset."""
    df = pd.read_csv(file_path)
    return df


def clean_url(url):
    """Clean and normalize URL."""
    # Remove protocol and www
    url = re.sub(r'https?://(www\.)?', '', url)
    # Remove trailing slashes
    url = url.rstrip('/')
    return url


def extract_url_features(url):
    """Extract structured features from URL."""
    try:
        parsed = urlparse(url)
    except Exception:
        parsed = urlparse("")

    features = {}

    # Domain features
    features['domain'] = parsed.netloc

    # Path features
    path_segments = parsed.path.strip('/').split('/') if parsed.path else []
    features['path_depth'] = len(path_segments)
    features['has_product_in_path'] = any(
        seg.lower() in ['product', 'products', 'item', 'items', 'dp', 'itm']
        for seg in path_segments
    )

    # Query parameters features
    query_params = parse_qs(parsed.query)
    features['num_query_params'] = len(query_params)
    features['has_id_param'] = any(
        'id' in key.lower() or 'pid' in key.lower()
        for key in query_params.keys()
    )

    # Pattern-based features
    features['has_product_pattern'] = bool(
        re.search(r'/(dp|product|item|itm)/[A-Z0-9]{6,}', url, re.I)
    )
    features['has_category_pattern'] = bool(
        re.search(r'/(category|categories|shop|store)/', url, re.I)
    )

    return features


def preprocess_data(df, label_mapping):
    """Main preprocessing function."""
    df_clean = df.copy()

    # Clean URLs
    df_clean['cleaned_url'] = df_clean['url'].apply(clean_url)

    # Extract URL features
    url_features = df_clean['url'].apply(extract_url_features).apply(pd.Series)
    df_clean = pd.concat([df_clean, url_features], axis=1)

    # Map labels if available
    if 'category' in df_clean.columns:
        df_clean['label'] = df_clean['category'].map(label_mapping).fillna('other')

    return df_clean


def split_data(df, test_size=0.2, random_state=42):
    """Split data into train and test sets."""
    return train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=df['label']
    )
