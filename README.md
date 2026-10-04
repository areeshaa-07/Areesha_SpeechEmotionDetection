# Speech Emotion Detection — Hexovate Assessment

## Project overview

A speech-emotion recognition system that predicts the emotional tone of
short speech clips in one of four classes: **angry, happy, sad, neutral**.
It is delivered as an interactive Streamlit dashboard that lets a reviewer
upload a WAV file or record directly from a microphone, and see the
predicted emotion, per-class probability bars, the input waveform, and a
results page with the model's evaluation on unseen speakers.

Use case: automatically flagging distressed or unhappy callers in a support
line, as described in the Hexovate assessment brief.

## Dataset

- **Source:** RAVDESS speech audio (Zenodo record 1188976 / Kaggle
  `uwrfkaggler/ravdess-emotional-speech-audio`).
- **Filtered** to the 4 target emotions → **672 clips** total:
  - angry: 192
  - happy: 192
  - sad: 192
  - **neutral: 96** (imbalanced by design — handled with class weights)
- **Emotion code** is the 3rd number in each RAVDESS filename:
  `01` neutral, `03` happy, `04` sad, `05` angry.

## Approach / methodology

1. **Exploration** — plotted clip counts per emotion and inspected one
   waveform + one mel spectrogram per emotion class.
2. **Feature extraction** — 40 MFCCs per clip via `librosa` at 22,050 Hz,
   mean-pooled over time → a 40-dimensional feature vector per clip.
3. **Actor-based split** — trained on actors **1–18**, tested on actors
   **19–24**. No speaker appears in both splits, so evaluation is on voices
   the model has never heard. Random clip-level splits were deliberately
   avoided because they leak speaker identity and inflate accuracy.
4. **Class imbalance** — the neutral class has half the samples of the
   others, so `class_weight='balanced'` was used in both candidate
   classifiers.
5. **Models compared** — Random Forest and SVM (RBF, with a
   `StandardScaler` pipeline).
6. **Evaluation** — accuracy, macro F1 and confusion matrix on the
   held-out actors.
7. **Deployment** — best model serialized with `joblib` and served through
   a Streamlit dashboard.

## Technologies and libraries used

| Tool | Purpose |
|---|---|
| Python 3.11 / 3.12 | Runtime |
| librosa | Audio loading, MFCC and mel-spectrogram features |
| NumPy | Numeric arrays |
| pandas | Dataframe handling and CSV export |
| scikit-learn | Random Forest, SVM, StandardScaler, metrics |
| joblib | Model serialization |
| matplotlib / seaborn | Waveform, spectrogram and confusion-matrix plots |
| Streamlit | Interactive dashboard and cloud deployment |
| Google Colab | Model training environment |
| GitHub | Source control and deployment trigger |

## Models used

- **Random Forest** (400 trees, `class_weight='balanced'`) — baseline.
- **SVM** (RBF kernel, `C=10`, `class_weight='balanced'`, wrapped in a
  `StandardScaler` pipeline) — deployed model.

## Results / evaluation

| Model | Accuracy | Macro F1 |
|---|---|---|
| Random Forest | 0.5060 | 0.4500 |
| **SVM (RBF, balanced)** | **0.5893** | **0.5725** |

The SVM wins on both metrics and is the deployed model. Evaluation is on
actors 19–24, who were never seen during training.

### Confusion analysis

Angry is by far the easiest class (39/48 correct) — high energy and low
spectral variance make it distinct. Sad is next (23/48). The dominant
confusions are:

- **Neutral → Sad (6/48)** and **Neutral → Happy (5/48)**. Neutral has half
  as many training clips and overlaps in mean-MFCC space with the other
  low-arousal classes.
- **Happy → Sad (13/48)**. Mean-pooling over time discards the pitch-contour
  and duration cues that separate happy from sad.

A CNN trained on mel spectrograms would preserve those temporal cues and is
the natural next step to reduce these errors.

### Known limitation — domain shift

The model is trained on **studio recordings of professional actors**
(close-mic, 48 kHz, controlled environment, strongly emoted). Live
microphone input from a laptop or phone is a very different acoustic
environment, and predictions on live audio are noticeably less confident
than on uploaded RAVDESS clips. This is expected and is the first thing I
would address with more time — for example by fine-tuning on in-the-wild
data or by training with channel and noise augmentation.

## Installation requirements

- Python **3.11** or **3.12** recommended (3.13+ may not have prebuilt
  wheels for `numba`, a librosa dependency).
- All Python dependencies are pinned in `requirements.txt`.

## How to run the project

```bash
git clone https://github.com/<areeshaa-07>/Areesha_SpeechEmotionDetection.git
cd Hamza_SpeechEmotionDetection

python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS / Linux:
source .venv/bin/activate

pip install -r requirements.txt
streamlit run app.py
```

Then open `http://localhost:8501`.

### Using the dashboard

1. **Predict tab** — upload a short WAV clip (or record one), and view the
   predicted emotion with per-class probabilities and the waveform.
2. **Results tab** — class counts, actor split, accuracy, macro F1 and the
   confusion matrix on the held-out speakers.

## Repository layout

```
app.py                      Streamlit dashboard
requirements.txt            Pinned dependencies
README.md                   This file
artifacts/
  svm_model.joblib          Trained SVM pipeline (scaler + SVC)
  metrics.json              Accuracy, F1, confusion matrix, class counts
sample_clips/               A few RAVDESS test clips for quick review
```

## Deployment

Live dashboard on Streamlit Community Cloud:
`<PASTE-LIVE-URL-HERE>`

## What I'd improve with more time

- Optional CNN on mel spectrograms, compared head-to-head with the SVM.
- Data augmentation (pitch shift, time stretch, background noise) to reduce
  overfitting to studio conditions and improve live-audio performance.
- Per-actor error analysis to check whether a few actors dominate the
  errors and whether the model is biased toward certain voice types.
- Probability calibration (Platt scaling / isotonic) so the confidence bars
  mean something absolute rather than just relative.
- Transfer learning from a pretrained audio model (wav2vec2, YAMNet) — much
  more likely to close the gap on live audio than training a CNN from
  scratch on 672 clips.

