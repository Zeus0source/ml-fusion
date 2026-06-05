import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
from PIL import Image
try:
    from tensorflow import keras
except ImportError:
   import keras
# Auto-setup: download models if they don't exist
if not os.path.exists("models/sentiment_model.pkl") or \
   not os.path.exists("models/cifar_model.h5"):
    import setup
    with st.spinner("Downloading models... please wait ~1 min"):
        setup.download_models()
    st.rerun()
# ── PAGE CONFIG (must be first) ────────────────────────────────────
st.set_page_config(page_title="ML Fusion", layout="wide")

# ── CSS ────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=VT323&family=Courier+Prime:wght@400;700&display=swap');

* { box-sizing: border-box; }

html, body, [data-testid="stAppViewContainer"] {
    background-color: #0a0a0a !important;
    color: #FFB000 !important;
    font-family: 'Courier Prime', monospace !important;
}

[data-testid="stSidebar"]    { display: none !important; }
[data-testid="stHeader"]     { display: none !important; }
[data-testid="stToolbar"]    { display: none !important; }
[data-testid="stDecoration"] { display: none !important; }
[data-testid="stMainMenu"]   { display: none !important; }

[data-testid="stMainBlockContainer"] {
    padding-top: 24px !important;
    padding-left: 48px !important;
    padding-right: 48px !important;
    max-width: 1200px !important;
    margin: 0 auto !important;
}

/* HEADINGS */
h1, h2, h3 {
    font-family: 'VT323', monospace !important;
    color: #FFB000 !important;
    letter-spacing: 3px !important;
}
h1 { font-size: 52px !important; }
h2 { font-size: 34px !important; }

hr { border-color: #FFB000 !important; opacity: 0.25; }

/* NAV BUTTONS */
[data-testid="stButton"] button {
    background: #0a0a0a !important;
    color: #FFB000 !important;
    border: 1px solid #FFB000 !important;
    border-radius: 0 !important;
    font-family: 'Courier Prime', monospace !important;
    font-size: 12px !important;
    letter-spacing: 2px !important;
    text-transform: uppercase !important;
    transition: all 0.15s !important;
    width: 100% !important;
}
[data-testid="stButton"] button:hover {
    background: #FFB000 !important;
    color: #0a0a0a !important;
}
[data-testid="stButton"] button[kind="primary"] {
    background: #FFB000 !important;
    color: #0a0a0a !important;
    font-weight: 700 !important;
}
[data-testid="stButton"] button[kind="primary"]:hover {
    background: #FFD700 !important;
}

/* INPUTS */
[data-testid="stTextArea"] textarea,
[data-testid="stTextInput"] input {
    background: #111 !important;
    color: #FFB000 !important;
    border: 1px solid #FFB000 !important;
    border-radius: 0 !important;
    font-family: 'Courier Prime', monospace !important;
    font-size: 14px !important;
}

/* METRICS */
[data-testid="stMetric"] {
    background: #111 !important;
    border: 1px solid #333 !important;
    padding: 16px !important;
}
[data-testid="stMetricValue"] {
    color: #FFB000 !important;
    font-family: 'VT323', monospace !important;
    font-size: 44px !important;
}
[data-testid="stMetricLabel"] {
    color: #886000 !important;
    font-size: 10px !important;
    letter-spacing: 2px !important;
    text-transform: uppercase !important;
}

/* FILE UPLOADER */
[data-testid="stFileUploader"] {
    border: 1px dashed #FFB000 !important;
    border-radius: 0 !important;
    background: #111 !important;
}

/* DATAFRAME */
[data-testid="stDataFrame"] { border: 1px solid #333 !important; }

/* PROGRESS BAR */
[data-testid="stProgress"] > div > div { background: #FFB000 !important; }

/* CAPTION */
[data-testid="stCaptionContainer"] {
    color: #886000 !important;
    font-family: 'Courier Prime', monospace !important;
}

/* EXPANDER */
[data-testid="stExpander"] {
    border: 1px solid #333 !important;
    border-radius: 0 !important;
    background: #111 !important;
}

/* PAGE HEADER */
.page-header {
    border-bottom: 1px solid #FFB000;
    padding-bottom: 16px;
    margin-bottom: 28px;
    margin-top: 8px;
}
.page-header .module-label {
    font-family: 'Courier Prime', monospace;
    font-size: 11px;
    color: #886000;
    letter-spacing: 4px;
    margin-bottom: 4px;
}
.page-header .page-title {
    font-family: 'VT323', monospace;
    font-size: 56px;
    color: #FFB000;
    letter-spacing: 4px;
    line-height: 1;
}
.page-header .page-sub {
    font-family: 'Courier Prime', monospace;
    font-size: 12px;
    color: #886000;
    letter-spacing: 1px;
    margin-top: 6px;
}

/* RETRO ALERTS */
.retro-alert {
    border: 1px solid #333;
    border-left: 3px solid #FFB000;
    background: #111;
    padding: 14px 18px;
    font-family: 'Courier Prime', monospace;
    font-size: 13px;
    color: #886000;
    line-height: 1.7;
    margin-bottom: 16px;
}
.retro-alert .label {
    font-size: 10px;
    letter-spacing: 3px;
    color: #554000;
    display: block;
    margin-bottom: 6px;
    text-transform: uppercase;
}
.retro-warn { border-left-color: #FF8C00; }

/* RETRO RESULTS */
.retro-result {
    border: 1px solid #333;
    background: #111;
    padding: 20px 24px;
    margin-bottom: 16px;
}
.retro-result .res-label {
    font-size: 10px;
    letter-spacing: 3px;
    color: #554000;
    display: block;
    margin-bottom: 8px;
    text-transform: uppercase;
    font-family: 'Courier Prime', monospace;
}
.retro-result .res-value {
    font-family: 'VT323', monospace;
    font-size: 56px;
    letter-spacing: 4px;
    display: block;
    line-height: 1;
}
.retro-result .res-conf {
    font-family: 'Courier Prime', monospace;
    font-size: 12px;
    color: #886000;
    margin-top: 8px;
    letter-spacing: 2px;
}
.positive .res-value { color: #00FF41; }
.negative .res-value { color: #FF4444; }

/* SCANLINES */
[data-testid="stAppViewContainer"]::after {
    content: '';
    position: fixed;
    top: 0; left: 0;
    width: 100%; height: 100%;
    pointer-events: none;
    background: repeating-linear-gradient(
        0deg, transparent, transparent 2px,
        rgba(0,0,0,0.04) 2px, rgba(0,0,0,0.04) 4px
    );
    z-index: 9998;
}

/* SCROLLBAR */
::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-track { background: #0a0a0a; }
::-webkit-scrollbar-thumb { background: #FFB000; }
 .page-header .page-sub {
    color: #AA7700 !important;
    font-weight: 1000 !important;
    letter-spacing: 1.5px !important;
}           
</style>
""", unsafe_allow_html=True)

# ── LOAD MODELS ────────────────────────────────────────────────────
@st.cache_resource
def load_sentiment_model():
    with open("models/sentiment_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("models/sentiment_vectorizer.pkl", "rb") as f:
        vectorizer = pickle.load(f)
    return model, vectorizer

@st.cache_resource
def load_image_model():
    model   = keras.models.load_model("models/cifar_model.h5")
    classes = np.load("models/class_names.npy")
    return model, classes

# ── SESSION STATE ──────────────────────────────────────────────────
if "page" not in st.session_state:
    st.session_state.page = "sentiment"
if "history" not in st.session_state:
    st.session_state.history = []
if "input_text" not in st.session_state:
    st.session_state.input_text = ""

# ── NAVBAR ─────────────────────────────────────────────────────────
def nav_button(label, key):
    if st.button(label, key=f"nav_{key}", use_container_width=True):
        st.session_state.page = key
        st.rerun()

st.markdown("""
<div style="
    border-bottom: 1px solid #FFB000;
    padding-bottom: 12px;
    margin-bottom: 20px;
    display: flex;
    align-items: center;
    justify-content: space-between;
">
    <span style="font-family: VT323, monospace; font-size: 28px; color: #FFB000; letter-spacing: 4px;">
        ML_FUSION
    </span>
</div>
""", unsafe_allow_html=True)

n1, n2, n3, n4 = st.columns(4)
with n1: nav_button("Sentiment", "sentiment")
with n2: nav_button("Image Classifier", "classifier")
with n3: nav_button("History", "history")
with n4: nav_button("About", "about")

st.markdown("<div style='margin-bottom:24px;'></div>", unsafe_allow_html=True)

page_key = st.session_state.page

# ══════════════════════════════════════════════════════════════════
# PAGE: SENTIMENT ANALYSIS
# ══════════════════════════════════════════════════════════════════
if page_key == "sentiment":

    st.markdown("""
    <div class="page-header">
        <div class="module-label">&gt; MODULE_01</div>
        <div class="page-title">SENTIMENT ANALYSIS</div>
        <div style="font-family:'Courier Prime',monospace;font-size:13px;color:#CC9900;font-weight:700;letter-spacing:1.5px;margin-top:6px;">
    Logistic Regression · TF-IDF · 91.21% accuracy · 50,000 IMDB reviews
</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="retro-alert">
        <span class="label">&gt; notice</span>
        Trained on positive/negative reviews only. Neutral or ambiguous text may give
        low-confidence results — always check the confidence score.
    </div>
    """, unsafe_allow_html=True)

    sentiment_model, vectorizer = load_sentiment_model()
    col1, col2 = st.columns([3, 2])

    with col1:
        st.subheader("Input")
        st.caption("Try an example:")
        ex1, ex2, ex3 = st.columns(3)
        examples = {
            "Positive": "This movie was absolutely brilliant. The performances were outstanding and the story was deeply moving. I loved every second of it!",
            "Negative": "Terrible film. Complete waste of time and money. The plot made no sense and the acting was dreadful.",
            "Neutral" : "It was okay. Some parts were interesting but overall nothing special. Probably won't watch it again."
        }
        if ex1.button("Positive", key="ex_pos"): st.session_state.input_text = examples["Positive"]
        if ex2.button("Negative", key="ex_neg"): st.session_state.input_text = examples["Negative"]
        if ex3.button("Neutral",  key="ex_neu"): st.session_state.input_text = examples["Neutral"]

        user_text = st.text_area(
            "Type or paste your text here:",
            value=st.session_state.input_text,
            height=180,
            placeholder="e.g. This movie was absolutely amazing..."
        )
        predict_btn = st.button("Analyse Sentiment", type="primary", use_container_width=True, key="analyse_btn")

    with col2:
        st.subheader("Result")
        if predict_btn and user_text.strip():
            vec        = vectorizer.transform([user_text])
            pred       = sentiment_model.predict(vec)[0]
            proba      = sentiment_model.predict_proba(vec)[0]
            pos_conf   = proba[list(sentiment_model.classes_).index("positive")] * 100
            neg_conf   = proba[list(sentiment_model.classes_).index("negative")] * 100
            confidence = max(pos_conf, neg_conf)
            css_class  = "positive" if pred == "positive" else "negative"
            symbol     = "▲ POSITIVE" if pred == "positive" else "▼ NEGATIVE"

            st.markdown(f"""
            <div class="retro-result {css_class}">
                <span class="res-label">&gt; prediction_output</span>
                <span class="res-value">{symbol}</span>
                <div class="res-conf">confidence: {confidence:.1f}%</div>
            </div>
            """, unsafe_allow_html=True)

            st.metric("Confidence", f"{confidence:.1f}%")
            st.markdown("**Probability breakdown:**")
            st.markdown(" Positive")
            st.progress(int(pos_conf))
            st.caption(f"{pos_conf:.1f}%")
            st.markdown(" Negative")
            st.progress(int(neg_conf))
            st.caption(f"{neg_conf:.1f}%")

            st.session_state.history.append({
                "Module"    : "Sentiment",
                "Input"     : user_text[:60] + "..." if len(user_text) > 60 else user_text,
                "Prediction": pred.capitalize(),
                "Confidence": f"{confidence:.1f}%"
            })


        elif predict_btn:
            st.markdown("""
            <div class="retro-alert retro-warn">
                <span class="label">&gt; warning</span>
                Please enter some text first.
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="retro-alert">
                <span class="label">&gt; waiting</span>
                Enter text on the left and click Analyse Sentiment.
            </div>
            """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════
# PAGE: IMAGE CLASSIFIER
# ══════════════════════════════════════════════════════════════════
elif page_key == "classifier":

    st.markdown("""
    <div class="page-header">
        <div class="module-label">&gt; MODULE_02</div>
        <div class="page-title">IMAGE CLASSIFIER</div>
        <div class="page-sub">CNN · TensorFlow/Keras · 83.44% accuracy · CIFAR-10 · 60,000 images</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="retro-alert">
        <span class="label">&gt; notice</span>
        Trained on CIFAR-10 — tiny 32×32 pixel images. High-resolution photos may give
        unexpected results due to distribution shift. For best results use simple centered
        images with plain backgrounds.
        Sample images: cs.toronto.edu/~kriz/cifar.html
    </div>
    """, unsafe_allow_html=True)

    st.markdown("**Categories:** airplane · automobile · bird · cat · deer · dog · frog · horse · ship · truck")
    st.markdown("---")

    image_model, class_names = load_image_model()
    col1, col2 = st.columns([2, 3])

    with col1:
        st.subheader("Upload Image")
        uploaded = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])
        if uploaded:
            img = Image.open(uploaded).convert("RGB")
            st.image(img, caption="Uploaded image", use_container_width=True)
            classify_btn = st.button("Classify Image", type="primary", use_container_width=True, key="classify_btn")
        else:
            st.markdown("""
            <div class="retro-alert">
                <span class="label">&gt; waiting</span>
                Upload a JPG or PNG image to classify.
            </div>
            """, unsafe_allow_html=True)
            classify_btn = False

    with col2:
        st.subheader("Result")
        if uploaded and classify_btn:
            img_resized = img.resize((32, 32))
            img_array   = np.array(img_resized).astype("float32") / 255.0
            img_batch   = img_array[np.newaxis, ...]
            predictions = image_model.predict(img_batch, verbose=0)[0]
            top3_idx    = predictions.argsort()[-3:][::-1]
            top_class   = class_names[top3_idx[0]]
            top_conf    = predictions[top3_idx[0]] * 100

            st.markdown(f"""
            <div class="retro-result positive">
                <span class="res-label">&gt; classification_output</span>
                <span class="res-value">{top_class.upper()}</span>
                <div class="res-conf">confidence: {top_conf:.1f}%</div>
            </div>
            """, unsafe_allow_html=True)

            st.metric("Confidence", f"{top_conf:.1f}%")
            st.markdown("**Top 3 predictions:**")
            chart_data = pd.DataFrame({
                "Class"      : [class_names[i] for i in top3_idx],
                "Probability": [predictions[i] * 100 for i in top3_idx]
            })
            st.bar_chart(chart_data.set_index("Class"))

            with st.expander("See all 10 class probabilities"):
                all_data = pd.DataFrame({
                    "Class"      : class_names,
                    "Probability": [f"{p*100:.2f}%" for p in predictions]
                }).sort_values("Probability", ascending=False)
                st.dataframe(all_data, use_container_width=True)

            st.session_state.history.append({
                "Module"    : "Image Classifier",
                "Input"     : uploaded.name,
                "Prediction": top_class.capitalize(),
                "Confidence": f"{top_conf:.1f}%"
            })
        elif not uploaded:
            st.markdown("""
            <div class="retro-alert">
                <span class="label">&gt; waiting</span>
                Upload an image on the left to see results here.
            </div>
            """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════
# PAGE: PREDICTION HISTORY
# ══════════════════════════════════════════════════════════════════
elif page_key == "history":

    st.markdown("""
    <div class="page-header">
        <div class="module-label">&gt; MODULE_03</div>
        <div class="page-title">PREDICTION HISTORY</div>
        <div class="page-sub">All predictions made in this session · exportable as CSV</div>
    </div>
    """, unsafe_allow_html=True)

    if st.session_state.history:
        df = pd.DataFrame(st.session_state.history)
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Predictions",     len(df))
        col2.metric("Sentiment Predictions", len(df[df["Module"]=="Sentiment"]))
        col3.metric("Image Predictions",     len(df[df["Module"]=="Image Classifier"]))
        st.markdown("---")
        st.dataframe(df, use_container_width=True)
        csv = df.to_csv(index=False)
        st.download_button(
            label="Download as CSV",
            data=csv,
            file_name="ml_fusion_history.csv",
            mime="text/csv"
        )
        if st.button("Clear History", key="clear_btn"):
            st.session_state.history = []
            st.rerun()
    else:
        st.markdown("""
        <div class="retro-alert">
            <span class="label">&gt; empty</span>
            No predictions yet. Go to Sentiment Analysis or Image Classifier to make some.
        </div>
        """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════
# PAGE: ABOUT
# ══════════════════════════════════════════════════════════════════
elif page_key == "about":

    st.markdown("""
    <div class="page-header">
        <div class="module-label">&gt; INFO</div>
        <div class="page-title">ABOUT</div>
        <div class="page-sub">ML Fusion — a dual-module machine learning web application</div>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Sentiment Analysis Model")
        st.markdown("""
- **Algorithm:** Logistic Regression
- **Features:** TF-IDF (50,000 features, bigrams)
- **Dataset:** IMDB Movie Reviews (50,000 samples)
- **Accuracy:** 91.21% on held-out test set
- **Library:** Scikit-learn
        """)
    with col2:
        st.subheader("Image Classification Model")
        st.markdown("""
- **Algorithm:** CNN (3 Conv blocks + 2 Dense layers)
- **Dataset:** CIFAR-10 (60,000 images, 10 classes)
- **Accuracy:** 83.44% on held-out test set
- **Library:** TensorFlow / Keras
        """)

    st.markdown("---")
    st.subheader("Tech Stack")
    st.markdown("`Python` · `TensorFlow` · `Keras` · `Scikit-learn` · `Pandas` · `NumPy` · `Streamlit` · `Pillow`")
    st.markdown("---")
    st.subheader("Built by")
    st.markdown("**Yash Rastogi**")
