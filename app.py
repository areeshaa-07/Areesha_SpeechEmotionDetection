import json, os
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import streamlit as st
import joblib

st.set_page_config(page_title="Emotion Detection", layout="wide")

ART = "artifacts"
SR = 22050
N_MFCC = 40
LABELS = ["angry", "happy", "sad", "neutral"]

@st.cache_resource
def load_model():
    return joblib.load(os.path.join(ART, "svm_model.joblib"))

@st.cache_data
def load_metrics():
    with open(os.path.join(ART, "metrics.json")) as f:
        return json.load(f)

model = load_model()
metrics = load_metrics()

def extract_mfcc_mean(y, sr=SR, n_mfcc=N_MFCC):
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    return mfcc.mean(axis=1).reshape(1, -1)

st.title("🎙️ Speech Emotion Detection")
st.caption("Upload a short WAV clip. Model: SVM (RBF) trained on RAVDESS MFCC means.")

tab_predict, tab_results = st.tabs(["Predict", "Results"])

# ---------------- TAB 1: PREDICT ----------------
with tab_predict:
    st.markdown("**Option 1 — upload a WAV file**")
    up = st.file_uploader("Upload a .wav file", type=["wav"])

    st.markdown("**Option 2 — record from your microphone**")
    rec = st.audio_input("Record a short clip (2–4 seconds, speak clearly)")
    st.caption("Note: live audio is recorded on a different mic and in a different "
           "acoustic environment than the training data. Predictions on live "
           "recordings are typically less confident than on uploaded RAVDESS clips.")

    # Prefer the upload if both are present
    audio_source = up if up is not None else rec

    if audio_source is not None:
        y, sr = librosa.load(audio_source, sr=SR)

        col1, col2 = st.columns([1, 1])

        with col1:
            st.subheader("Waveform")
            fig, ax = plt.subplots(figsize=(6, 3))
            librosa.display.waveshow(y, sr=sr, ax=ax)
            ax.set_xlabel("Time (s)")
            st.pyplot(fig)

        with col2:
            st.subheader("Prediction")
            feats = extract_mfcc_mean(y, sr)
            proba = model.predict_proba(feats)[0]
            pred = model.classes_[int(np.argmax(proba))]

            st.markdown(f"### Predicted emotion: **{pred.upper()}**")
            for label, p in sorted(zip(model.classes_, proba),
                                   key=lambda x: -x[1]):
                st.write(f"**{label}** — {p*100:.1f}%")
                st.progress(float(p))

            st.caption(f"Clip length: {len(y)/sr:.2f}s  •  Sample rate: {sr} Hz")

# ---------------- TAB 2: RESULTS ----------------
with tab_results:
    st.subheader("Dataset & Evaluation")

    st.markdown("**Class counts (whole dataset after filtering):**")
    counts = metrics["class_counts"]
    st.bar_chart({k: counts[k] for k in LABELS})

    st.markdown(
        f"**Actor split:** train = actors {metrics['actor_split']['train'][0]}–"
        f"{metrics['actor_split']['train'][-1]}, "
        f"test = actors {metrics['actor_split']['test'][0]}–"
        f"{metrics['actor_split']['test'][-1]}"
    )

    c1, c2, c3 = st.columns(3)
    c1.metric("Best model", metrics["best_model"].upper())
    c2.metric("Accuracy", f"{metrics['accuracy']:.4f}")
    c3.metric("Macro F1", f"{metrics['macro_f1']:.4f}")

    st.markdown("**Confusion matrix (rows = true, cols = predicted)**")
    cm = np.array(metrics["confusion_matrix"])
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(LABELS))); ax.set_xticklabels(LABELS)
    ax.set_yticks(range(len(LABELS))); ax.set_yticklabels(LABELS)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    for i in range(len(LABELS)):
        for j in range(len(LABELS)):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color="white" if cm[i, j] > cm.max()/2 else "black")
    st.pyplot(fig)
