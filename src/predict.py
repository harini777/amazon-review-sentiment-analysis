"""Predict sentiment for a custom review."""

import pickle

from preprocess import clean_text


MODEL_PATH = "sentiment_model.pkl"
VECTORIZER_PATH = "tfidf_vectorizer.pkl"


def predict_sentiment(review):
    """Return positive or negative sentiment for one review."""
    with open(MODEL_PATH, "rb") as model_file:
        model = pickle.load(model_file)

    with open(VECTORIZER_PATH, "rb") as vectorizer_file:
        vectorizer = pickle.load(vectorizer_file)

    cleaned_review = clean_text(review)
    review_vector = vectorizer.transform([cleaned_review])
    prediction = model.predict(review_vector)

    return prediction[0]


if __name__ == "__main__":
    sample_review = "This product is amazing and I loved it"
    print("Review:", sample_review)
    print("Predicted Sentiment:", predict_sentiment(sample_review))
