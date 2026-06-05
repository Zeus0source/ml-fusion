#  ML Fusion

A dual-module machine learning web application built with Python and Streamlit.

 **Live Demo:** [yashr06-ml-fusion.hf.space](https://yashr06-ml-fusion.hf.space)

---

## Modules

**Sentiment Analysis**
- Algorithm: Logistic Regression with TF-IDF (bigrams, 50k features)
- Dataset: IMDB Movie Reviews (50,000 samples)
- Accuracy: 91.21% on held-out test set

**Image Classifier**
- Algorithm: CNN (3 Conv blocks + 2 Dense layers)
- Dataset: CIFAR-10 (60,000 images, 10 classes)
- Accuracy: 83.44% on held-out test set

---

## Tech Stack

`Python` · `TensorFlow` · `Keras` · `Scikit-learn` · `Pandas` · `NumPy` · `Streamlit` · `Hugging Face`

---

## Run Locally

```bash
git clone https://github.com/Zeus0source/ml-fusion
cd ml-fusion
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```
---

## Project Structure
ml-fusion/

├── app.py                 # Main Streamlit app

├── setup.py               # Model download from Hugging Face

├── train_sentiments.py    # Sentiment model training script

├── train_cifar.py         # CNN training script

└── requirements.txt
