"""
preprocessing.py — Text cleaning pipeline for the Fake News Detector.

Implements a custom sklearn-compatible transformer that cleans raw
article text into a normalized form suitable for TF-IDF vectorization.

Cleaning steps:
1. Lowercase all text
2. Remove URLs
3. Remove HTML tags
4. Remove special characters and digits
5. Tokenize
6. Remove English stopwords
7. Lemmatize each token
8. Rejoin into a single cleaned string

Run independently:
    python3 src/preprocessing.py

This will demonstrate the cleaning on sample text.
"""

import re
import nltk
from sklearn.base import BaseEstimator, TransformerMixin

# Download required NLTK data (only on first run)
nltk.download("stopwords", quiet=True)
nltk.download("wordnet", quiet=True)
nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


class TextPreprocessor(BaseEstimator, TransformerMixin):
    """
    Sklearn-compatible text preprocessing transformer.

    Fits into an sklearn Pipeline so the same cleaning is applied
    during both training and inference — preventing training-serving skew.

    Usage:
        preprocessor = TextPreprocessor()
        cleaned_texts = preprocessor.transform(["Some raw article text..."])
    """

    def __init__(self):
        self.stop_words = set(stopwords.words("english"))
        self.lemmatizer = WordNetLemmatizer()

    def fit(self, X, y=None):
        """No fitting needed — this is a stateless transformer."""
        return self

    def transform(self, X, y=None):
        """
        Clean a list/series of raw text strings.

        Args:
            X: iterable of strings (raw article text)

        Returns:
            list of cleaned strings
        """
        return [self._clean_text(text) for text in X]

    def _clean_text(self, text):
        """
        Apply the full cleaning pipeline to a single text string.

        Args:
            text: raw article text (string)

        Returns:
            cleaned text (string)
        """
        # Handle non-string inputs
        if not isinstance(text, str):
            return ""

        # Step 1: Lowercase
        text = text.lower()

        # Step 2: Remove URLs
        text = re.sub(r"http\S+|www\.\S+", "", text)

        # Step 3: Remove HTML tags
        text = re.sub(r"<.*?>", "", text)

        # Step 4: Remove special characters and digits
        text = re.sub(r"[^a-zA-Z\s]", "", text)

        # Step 5: Tokenize (simple whitespace split)
        tokens = text.split()

        # Step 6: Remove stopwords
        tokens = [t for t in tokens if t not in self.stop_words]

        # Step 7: Lemmatize
        tokens = [self.lemmatizer.lemmatize(t) for t in tokens]

        # Step 8: Rejoin
        return " ".join(tokens)


if __name__ == "__main__":
    print("=" * 60)
    print("Text Preprocessing Demo")
    print("=" * 60)

    sample = [
        "BREAKING: Scientists discover <b>new</b> species! Visit http://example.com for more.",
        "The president said 123 things about the economy today...",
        "EXPOSED!! Secret government plot to control YOUR mind!!! Share before DELETED!!!",
    ]

    preprocessor = TextPreprocessor()
    cleaned = preprocessor.transform(sample)

    for i, (original, clean) in enumerate(zip(sample, cleaned), 1):
        print(f"\n  Sample {i}:")
        print(f"    Original: {original}")
        print(f"    Cleaned:  {clean}")

    print(f"\n✓ Preprocessor is working correctly")
