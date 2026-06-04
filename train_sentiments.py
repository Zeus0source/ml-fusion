# ─── STEP 1: IMPORTS ───────────────────────────────────────────────
# pandas  → loads our CSV into a table structure called a DataFrame
# sklearn → our ML toolkit. We use 4 specific parts:
#     TfidfVectorizer    → converts raw text into numbers
#     LogisticRegression → the classification model we'll train
#     train_test_split   → splits data so we can test fairly
#     accuracy_score,
#     classification_report → measures how well our model performs
# pickle → saves the trained model to a file so Streamlit can reuse it
# os     → used to create the models/ folder if it doesn't exist yet

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import pickle
import os

# ─── STEP 2: LOAD THE DATASET ──────────────────────────────────────
# We read the CSV into a DataFrame called df.
# A DataFrame is like an Excel sheet in Python —
# rows are individual reviews, columns are 'review' and 'sentiment'.
# We then print the shape (rows, columns) and first 3 rows to verify
# everything loaded correctly before doing anything else.

print("\n--- Step 1: Loading dataset ---")
df = pd.read_csv("data/IMDB Dataset.csv")
print(f"Shape: {df.shape}")
print(df.head(3))

# ─── STEP 3: PREPARE X AND y ───────────────────────────────────────
# In ML, we always split data into:
#   X → the INPUT  (what the model reads)   = the review text
#   y → the OUTPUT (what the model predicts) = positive or negative
#
# We also do a small cleanup:
#   .str.lower() → makes all text lowercase so "Good" and "good" are treated the same
#   .str.strip() → removes accidental spaces at start/end of labels
import re

def clean_text(text):
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = text.lower().strip()
    return text

print("\n--- Step 2: Preparing inputs and labels ---")
X = df["review"].apply(clean_text)
y = df["sentiment"].str.lower().str.strip()

print(f"Total reviews : {len(X)}")
print(f"Label counts  :\n{y.value_counts()}")

# ─── STEP 4: TRAIN / TEST SPLIT ────────────────────────────────────
# We can't test a model on the same data it learned from —
# that's like giving students the exam answers while studying.
# train_test_split() shuffles and splits our data:
#   80% goes to training  → the model learns from this
#   20% goes to testing   → we use this ONLY to measure final accuracy
#
# random_state=42 just fixes the shuffle so results are reproducible.
# (42 is a convention in ML — any number works.)

print("\n--- Step 3: Splitting into train and test sets ---")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"Training samples : {len(X_train)}")
print(f"Testing  samples : {len(X_test)}")

# ─── STEP 5: TF-IDF VECTORIZATION ──────────────────────────────────
# ML models can't understand raw words — they only understand numbers.
# TF-IDF (Term Frequency–Inverse Document Frequency) converts each
# review into a row of numbers, one number per unique word.
#
# TF  = how often a word appears in THIS review
# IDF = how rare that word is across ALL reviews
# Words like "the", "is", "a" appear everywhere → low IDF → low score
# Words like "brilliant", "terrible" are rarer → high IDF → high score
#
# max_features=10000 → we only keep the top 10,000 most useful words
# (ignoring ultra-rare words that appear in only 1-2 reviews)
#
# IMPORTANT: we call .fit_transform() on TRAINING data only.
# Then .transform() on test data.
# This prevents "data leakage" — the vectorizer must not know
# anything about the test set when it's learning the vocabulary.

print("\n--- Step 4: Converting text to numbers (TF-IDF) ---")
vectorizer = TfidfVectorizer(max_features=50000,ngram_range=(1, 2))
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec  = vectorizer.transform(X_test)

print(f"Each review is now a row of {X_train_vec.shape[1]} numbers")

# ─── STEP 6: TRAIN THE MODEL ───────────────────────────────────────
# LogisticRegression is a classification algorithm.
# Despite the name it's used for CLASSIFICATION not regression.
# It learns a weight for each of our 10,000 words.
# Words like "excellent", "loved", "amazing" get positive weights.
# Words like "awful", "boring", "waste" get negative weights.
# For a new review it multiplies word scores × weights and
# outputs a probability: e.g. 92% positive, 8% negative.
#
# max_iter=1000 → gives the algorithm enough iterations to converge.
# C=1.0 → controls regularization (prevents overfitting). Default is fine.

print("\n--- Step 5: Training the model ---")
model = LogisticRegression(max_iter=1000, C=5.0)
model.fit(X_train_vec, y_train)
print("Training complete!")

# ─── STEP 7: EVALUATE THE MODEL ────────────────────────────────────
# Now we test on the 20% the model has NEVER seen.
# accuracy_score → what % of predictions were correct overall
# classification_report → breaks it down by class:
#   precision → of all reviews predicted positive, how many actually were?
#   recall    → of all actual positives, how many did we catch?
#   f1-score  → harmonic mean of precision and recall (overall quality)

print("\n--- Step 6: Evaluating on test set ---")
y_pred = model.predict(X_test_vec)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")
print("\nDetailed report:")
print(classification_report(y_test, y_pred))

# ─── STEP 8: SAVE THE MODEL AND VECTORIZER ─────────────────────────
# We save both the model AND the vectorizer using pickle.
# pickle serializes a Python object into a binary file (.pkl).
#
# WHY save the vectorizer too?
# When a user types a review in Streamlit, we need to convert THEIR
# text into numbers using the EXACT SAME vocabulary the model was
# trained on. If we create a new vectorizer, the word→number mapping
# will be different and predictions will be garbage.
# So both files must always travel together.

print("\n--- Step 7: Saving model and vectorizer ---")
os.makedirs("models", exist_ok=True)

with open("models/sentiment_model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("models/sentiment_vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)

print("Saved: models/sentiment_model.pkl")
print("Saved: models/sentiment_vectorizer.pkl")

# ─── STEP 9: QUICK SANITY CHECK ────────────────────────────────────
# Before finishing, let's manually test 3 reviews to make sure
# the saved model actually works end-to-end.
# We load the files fresh from disk (not from memory) to simulate
# exactly what Streamlit will do later.

print("\n--- Step 8: Quick sanity check ---")
with open("models/sentiment_model.pkl", "rb") as f:
    loaded_model = pickle.load(f)

with open("models/sentiment_vectorizer.pkl", "rb") as f:
    loaded_vectorizer = pickle.load(f)

test_reviews = [
    "This movie was absolutely brilliant. I loved every second of it!",
    "Terrible film. Complete waste of time and money. Very boring.",
    "It was okay, nothing special but not bad either."
]

for review in test_reviews:
    vec   = loaded_vectorizer.transform([review])
    pred  = loaded_model.predict(vec)[0]
    proba = loaded_model.predict_proba(vec)[0]
    conf  = max(proba) * 100
    print(f"\nReview : {review[:55]}...")
    print(f"Result : {pred.upper()} ({conf:.1f}% confidence)")

    