import os
os.environ["STREAMLIT_DISABLE_USAGE_STATS"] = "true"
os.environ["STREAMLIT_HOME"] = "/tmp/.streamlit"
os.makedirs(os.environ["STREAMLIT_HOME"], exist_ok=True)

import streamlit as st
import joblib
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import (
    confusion_matrix, roc_curve, auc, precision_score,
    recall_score, f1_score, accuracy_score
)
from sentence_transformers import SentenceTransformer

# ------------------ Load Saved Model & Vectorizer ------------------
@st.cache_resource
def load_model():
    model = joblib.load("resume_match_model.pkl")
    tfidf = joblib.load("tfidf_vectorizer.pkl")
    sbert = SentenceTransformer("all-mpnet-base-v2")
    return model, tfidf, sbert

model, tfidf, sbert = load_model()

# ------------------ Streamlit App Layout ------------------
st.set_page_config(page_title="AI Resume-Job Matcher", page_icon="📄", layout="wide")
st.title("📄 AI-Powered Resume–Job Matching System")

# ------------------ Tabs ------------------
tab1, tab2 = st.tabs(["🎯 Live Prediction", "📊 Model Performance Dashboard"])

# ------------------ TAB 1: Prediction ------------------
with tab1:
    st.header("🎯 Predict Resume-Job Compatibility")
    st.write("Paste your **resume text** and **job description** to get a **match score**.")

    resume_text = st.text_area("📄 Resume Text", height=200)
    job_text = st.text_area("💼 Job Description", height=200)

    if st.button("🔍 Predict Match Score"):
        if not resume_text or not job_text:
            st.warning("⚠️ Please enter both Resume and Job Description!")
        else:
            # SBERT Embeddings
            resume_emb = sbert.encode([resume_text])
            job_emb = sbert.encode([job_text])

            # TF-IDF Features
            resume_tfidf = tfidf.transform([resume_text]).toarray()
            job_tfidf = tfidf.transform([job_text]).toarray()

            # Combine Features
            features = np.hstack([resume_emb, job_emb, resume_tfidf, job_tfidf])

            # Prediction
            match_prob = model.predict_proba(features)[0][1] * 100

            # Display Result
            st.subheader(f"🎯 Match Score: **{match_prob:.2f}%**")
            if match_prob >= 70:
                st.success("✅ Strong Match! Your resume is highly relevant.")
                st.balloons()
            elif match_prob >= 40:
                st.warning("⚠️ Moderate Match — You may need to improve your resume.")
            else:
                st.error("❌ Weak Match — Consider updating your resume.")

# ------------------ TAB 2: Dashboard ------------------
with tab2:
    st.header("📊 Model Performance Dashboard")

    # Sample evaluation metrics (replace with your actual optimized model results)
    metrics_data = {
        "Model": ["Baseline", "Optimized"],
        "Accuracy": [0.65, 0.83],
        "Precision": [0.66, 0.82],
        "Recall": [0.70, 0.85],
        "F1 Score": [0.68, 0.83],
        "ROC-AUC": [0.77, 0.90]
    }
    df_metrics = pd.DataFrame(metrics_data)

    st.subheader("📌 Before vs After Model Comparison")
    st.dataframe(df_metrics, use_container_width=True)

    # Visual Comparison
    st.subheader("📈 Performance Metrics Comparison")
    fig, ax = plt.subplots(figsize=(8, 5))
    df_metrics.set_index("Model")[["Accuracy", "Precision", "Recall", "F1 Score", "ROC-AUC"]].plot(kind="bar", ax=ax)
    plt.title("Before vs After Model Performance")
    plt.ylabel("Score")
    plt.ylim(0, 1)
    st.pyplot(fig)

    

    # ROC Curve 
    st.subheader("📌 ROC Curve — Optimized Model")
    y_scores = np.array([0.2, 0.4, 0.8, 0.9, 0.3, 0.1, 0.85, 0.65])  # Example (replace with real scores)
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)
    fig, ax = plt.subplots()
    plt.plot(fpr, tpr, color="blue", lw=2, label=f"AUC = {roc_auc:.2f}")
    plt.plot([0, 1], [0, 1], color="gray", linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend(loc="lower right")

    st.pyplot(fig)
