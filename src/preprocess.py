"""Simple preprocessing functions for sentiment analysis."""

import re

from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS


stemmer = PorterStemmer()
stop_words = set(ENGLISH_STOP_WORDS)

# Keep negation words because they are important for sentiment.
negation_words = {"no", "not", "never", "nor"}
stop_words = stop_words - negation_words


def get_sentiment(score):
    """Convert review rating into positive, negative, or neutral."""
    if score >= 4:
        return "positive"
    if score <= 2:
        return "negative"
    return "neutral"


def clean_text(text):
    """Clean review text before converting it into TF-IDF features."""
    text = str(text).lower()
    text = re.sub(r"[^a-zA-Z]", " ", text)
    words = text.split()
    words = [stemmer.stem(word) for word in words if word not in stop_words]
    return " ".join(words)


def prepare_data(df):
    """Create sentiment labels and cleaned review text."""
    df = df.dropna(subset=["Text", "Score"]).copy()
    df["sentiment"] = df["Score"].apply(get_sentiment)
    df = df[df["sentiment"] != "neutral"]
    df["clean_text"] = df["Text"].apply(clean_text)
    return df
