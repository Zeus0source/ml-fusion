import os
import pickle
import numpy as np
import re

def download_imdb():
    if os.path.exists("data/IMDB Dataset.csv"):
        print("IMDB dataset already exists, skipping.")
        return

    print("Downloading IMDB dataset from Kaggle...")
    import streamlit as st
    os.environ["KAGGLE_USERNAME"] = st.secrets["kaggle"]["username"]
    os.environ["KAGGLE_KEY"]      = st.secrets["kaggle"]["key"]

    import kaggle
    os.makedirs("data", exist_ok=True)
    kaggle.api.authenticate()
    kaggle.api.dataset_download_files(
        "lakshmi25npathi/imdb-dataset-of-50k-movie-reviews",
        path="data/",
        unzip=True
    )
    print("IMDB dataset downloaded.")

def train_sentiment():
    if os.path.exists("models/sentiment_model.pkl"):
        print("Sentiment model already exists, skipping.")
        return

    download_imdb()

    print("Training sentiment model...")
    import pandas as pd
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split

    df = pd.read_csv("data/IMDB Dataset.csv")

    def clean_text(text):
        text = re.sub(r'<.*?>', '', text)
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        return text.lower().strip()

    X = df["review"].apply(clean_text)
    y = df["sentiment"].str.lower().str.strip()

    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42)

    vectorizer = TfidfVectorizer(max_features=50000, ngram_range=(1,2), sublinear_tf=True)
    X_train_vec = vectorizer.fit_transform(X_train)

    model = LogisticRegression(max_iter=1000, C=5.0)
    model.fit(X_train_vec, y_train)

    os.makedirs("models", exist_ok=True)
    with open("models/sentiment_model.pkl", "wb") as f:
        pickle.dump(model, f)
    with open("models/sentiment_vectorizer.pkl", "wb") as f:
        pickle.dump(vectorizer, f)
    print("Sentiment model trained and saved.")

def train_cifar():
    if os.path.exists("models/cifar_model.h5"):
        print("CIFAR model already exists, skipping.")
        return

    print("Training CIFAR model...")
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    from tensorflow.keras.datasets import cifar10

    (X_train, y_train), (X_test, y_test) = cifar10.load_data()
    X_train = X_train.astype("float32") / 255.0
    X_test  = X_test.astype("float32")  / 255.0

    model = keras.Sequential([
        layers.Conv2D(32, (3,3), activation="relu", padding="same", input_shape=(32,32,3)),
        layers.BatchNormalization(),
        layers.Conv2D(32, (3,3), activation="relu", padding="same"),
        layers.BatchNormalization(),
        layers.MaxPooling2D(2,2),
        layers.Dropout(0.25),
        layers.Conv2D(64, (3,3), activation="relu", padding="same"),
        layers.BatchNormalization(),
        layers.Conv2D(64, (3,3), activation="relu", padding="same"),
        layers.BatchNormalization(),
        layers.MaxPooling2D(2,2),
        layers.Dropout(0.25),
        layers.Conv2D(128, (3,3), activation="relu", padding="same"),
        layers.BatchNormalization(),
        layers.MaxPooling2D(2,2),
        layers.Dropout(0.25),
        layers.Flatten(),
        layers.Dense(256, activation="relu"),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(10, activation="softmax")
    ])

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    model.fit(X_train, y_train, epochs=15, batch_size=64,
              validation_data=(X_test, y_test), verbose=1)

    os.makedirs("models", exist_ok=True)
    model.save("models/cifar_model.h5")
    CLASS_NAMES = ["airplane","automobile","bird","cat","deer",
                   "dog","frog","horse","ship","truck"]
    np.save("models/class_names.npy", CLASS_NAMES)
    print("CIFAR model trained and saved.")

if __name__ == "__main__":
    train_sentiment()
    train_cifar()