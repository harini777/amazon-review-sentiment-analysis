"""Train the sentiment analysis model."""

import pickle
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

from preprocess import prepare_data


DATA_PATH = "data/Reviews.csv"
MODEL_PATH = "sentiment_model.pkl"
VECTORIZER_PATH = "tfidf_vectorizer.pkl"

if not Path(DATA_PATH).exists():
    print("Dataset not found.")
    print("Please place the file here: data/Reviews.csv")
    raise SystemExit

df = pd.read_csv(DATA_PATH)

# Use a smaller sample so the project runs quickly on normal laptops.
df = df.sample(30000, random_state=42)
df = prepare_data(df)

print("Before balancing:")
print(df["sentiment"].value_counts())

# Balance positive and negative reviews so the model does not predict only positive.
positive_reviews = df[df["sentiment"] == "positive"]
negative_reviews = df[df["sentiment"] == "negative"]
sample_size = min(len(positive_reviews), len(negative_reviews))

positive_reviews = positive_reviews.sample(sample_size, random_state=42)
negative_reviews = negative_reviews.sample(sample_size, random_state=42)

df = pd.concat([positive_reviews, negative_reviews])
df = df.sample(frac=1, random_state=42)

print("\nAfter balancing:")
print(df["sentiment"].value_counts())

vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(df["clean_text"])
y = df["sentiment"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y,
)

model = MultinomialNB()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n")
print(confusion_matrix(y_test, y_pred))

with open(MODEL_PATH, "wb") as model_file:
    pickle.dump(model, model_file)

with open(VECTORIZER_PATH, "wb") as vectorizer_file:
    pickle.dump(vectorizer, vectorizer_file)

print("\nModel and vectorizer saved successfully.")
