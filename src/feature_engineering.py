"""
feature_engineering.py — TF-IDF vectorizer configuration.

Converts cleaned text into numerical feature vectors using
Term Frequency–Inverse Document Frequency (TF-IDF).

Key hyperparameters:
- max_features=50000: Limits vocabulary to top 50K terms by frequency
- ngram_range=(1, 2): Captures both unigrams and bigrams
- min_df=2: Ignores terms that appear in fewer than 2 documents
- max_df=0.95: Ignores terms that appear in more than 95% of documents
- sublinear_tf=True: Applies log normalization to term frequencies
"""

from sklearn.feature_extraction.text import TfidfVectorizer


def get_tfidf_vectorizer():
    """
    Create and return a configured TF-IDF vectorizer.

    The settings are tuned for news article classification:
    - Bigrams help capture phrases like "fake news", "breaking news"
    - Sublinear TF reduces the impact of very frequent terms
    - min_df/max_df filter out noise (too rare or too common terms)

    Returns:
        sklearn.feature_extraction.text.TfidfVectorizer
    """
    vectorizer = TfidfVectorizer(
        max_features=50000,
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95,
        sublinear_tf=True,
        strip_accents="unicode",
        dtype="float32",  # Save memory with float32 instead of float64
    )

    return vectorizer


if __name__ == "__main__":
    # Quick test
    vectorizer = get_tfidf_vectorizer()
    print(f"TF-IDF Vectorizer config:")
    print(f"  max_features: {vectorizer.max_features}")
    print(f"  ngram_range:  {vectorizer.ngram_range}")
    print(f"  min_df:       {vectorizer.min_df}")
    print(f"  max_df:       {vectorizer.max_df}")
    print(f"  sublinear_tf: {vectorizer.sublinear_tf}")

    # Test with sample data
    sample = ["this is a test document", "another test document here"]
    X = vectorizer.fit_transform(sample)
    print(f"\n  Sample output shape: {X.shape}")
    print(f"  Feature names (first 10): {vectorizer.get_feature_names_out()[:10].tolist()}")
