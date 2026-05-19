"""Simple Streamlit app for sentiment prediction."""

import sys
from pathlib import Path

import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parent
SRC_PATH = PROJECT_ROOT / "src"
sys.path.append(str(SRC_PATH))

from predict import predict_sentiment  # noqa: E402


st.set_page_config(
    page_title="Amazon Review Sentiment Analyzer",
    page_icon=":bar_chart:",
    layout="centered",
)

st.title("Amazon Review Sentiment Analyzer")
st.write("Enter a food product review and check whether it is positive or negative.")

review = st.text_area(
    "Review text",
    placeholder="Example: This product tasted amazing and I would buy it again.",
    height=150,
)

if st.button("Analyze Sentiment", type="primary"):
    if not review.strip():
        st.warning("Please enter a review before analyzing.")
    else:
        try:
            sentiment = predict_sentiment(review)
            if sentiment == "positive":
                st.success("Predicted Sentiment: Positive")
            else:
                st.error("Predicted Sentiment: Negative")
        except FileNotFoundError:
            st.error("Model files not found. Please train the model first.")
            st.code("python src/train_model.py")
