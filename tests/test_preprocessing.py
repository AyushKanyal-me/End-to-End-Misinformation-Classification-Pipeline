import pytest
from src.preprocessing import TextPreprocessor

@pytest.fixture
def preprocessor():
    return TextPreprocessor()

def test_lowercase(preprocessor):
    cleaned = preprocessor.transform(["SHOUTING"])
    assert cleaned[0] == "shouting"

def test_remove_urls(preprocessor):
    cleaned = preprocessor.transform(["Visit http://example.com today!"])
    assert "http" not in cleaned[0]
    
def test_remove_html(preprocessor):
    cleaned = preprocessor.transform(["<b>bold</b>"])
    assert "bold" in cleaned[0]
    assert "<b>" not in cleaned[0]

def test_stopwords_and_lemmatization(preprocessor):
    cleaned = preprocessor.transform(["The bats are flying over the buildings"])
    # 'the' 'are' 'over' should be removed
    # 'bats' -> 'bat', 'buildings' -> 'building'
    assert "bat flying building" in cleaned[0] or "bat flying" in cleaned[0]
