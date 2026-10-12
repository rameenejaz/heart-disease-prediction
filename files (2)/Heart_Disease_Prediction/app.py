"""
Heart Disease Prediction System - Web App (Streamlit)

Loads the trained SVC model and label encoders saved by the notebook and
predicts whether a patient has heart disease from the 4 selected attributes.

Run locally:   streamlit run app.py
"""

import pickle
from pathlib import Path

import pandas as pd
import streamlit as st

BASE_DIR = Path(__file__).parent

st.set_page_config(page_title="Heart Disease Prediction", page_icon="🫀", layout="centered")

st.markdown(
    """
    <style>
    .block-container { max-width: 760px; padding-top: 2.5rem; }
    h1 { font-weight: 700; letter-spacing: -0.01em; }
    .result {
        border-radius: 10px; padding: 1.2rem 1.4rem; margin-top: 0.5rem;
        border: 1px solid; font-size: 1.05rem; line-height: 1.5; margin-bottom: 1rem;
    }
    .result strong { display: block; font-size: 1.45rem; margin-bottom: 0.2rem; }
    .result.yes { background: #FDF1F0; border-color: #E8B4AE; color: #8A1C12; }
    .result.no  { background: #EEF6F1; border-color: #A9D0B8; color: #1E5631; }
    .note { color: #5B6573; font-size: 0.85rem; margin-top: 2rem; }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_artifacts():
    """Load the trained model and the label encoders once and reuse them."""
    with open(BASE_DIR / "heart_disease_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open(BASE_DIR / "heart_disease_label_encoders.pkl", "rb") as f:
        encoders = pickle.load(f)
    return model, encoders


try:
    model, encoders = load_artifacts()
except FileNotFoundError as error:
    st.error(
        f"Model file not found: {Path(error.filename).name}. "
        "Keep heart_disease_model.pkl and heart_disease_label_encoders.pkl in the same folder as app.py."
    )
    st.stop()

st.title("Heart Disease Prediction")
st.write(
    "Enter four clinical findings for a patient. A Support Vector Machine trained on the "
    "UCI Cleveland dataset predicts whether heart disease is present."
)

col1, col2 = st.columns(2)
with col1:
    chest_pain = st.selectbox(
        "Chest pain type",
        ["Typical Angina", "Atypical Angina", "Non-Anginal Pain", "Asymptomatic"],
        help="Type of chest pain the patient reports (cp).",
    )
    exercise_angina = st.selectbox(
        "Chest pain during exercise",
        ["No", "Yes"],
        help="Exercise-induced angina (exang).",
    )
with col2:
    major_vessels = st.selectbox(
        "Major vessels seen on fluoroscopy",
        ["Zero", "One", "Two", "Three"],
        help="Number of major vessels colored by fluoroscopy (ca).",
    )
    thal = st.selectbox(
        "Thallium stress test result",
        ["Normal", "Fixed Defect", "Reversible Defect"],
        help="Result of the thallium heart scan (thal).",
    )

if st.button("Predict", type="primary", use_container_width=True):
    # Build the feature vector in the same column order used for training
    patient = pd.DataFrame({
        "ChestPain": [chest_pain],
        "MajorVessels": [major_vessels],
        "ExerciseAngina": [exercise_angina],
        "Thal": [thal],
    })

    # Encode exactly as during training
    encoded = patient.copy()
    for column in encoded.columns:
        encoded[column] = encoders[column].transform(patient[column])

    prediction = model.predict(encoded)[0]

    if prediction == 1:
        st.markdown(
            '<div class="result yes"><strong>Heart disease: Yes</strong>'
            "The model predicts heart disease is present for these findings.</div>",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<div class="result no"><strong>Heart disease: No</strong>'
            "The model predicts no heart disease for these findings.</div>",
            unsafe_allow_html=True,
        )

with st.expander("About the model"):
    st.markdown(
        """
- **Algorithm:** Support Vector Classifier (scikit-learn)
- **Dataset:** UCI Heart Disease, Cleveland subset (297 patients after cleaning)
- **Features:** the 4 most discriminating attributes, chosen with the Chi-Square test and Mutual Information
- **Test accuracy:** 0.85 &nbsp;|&nbsp; **5-fold cross-validation accuracy:** 0.79
        """
    )

st.markdown(
    '<p class="note">Educational project. Not a substitute for medical diagnosis.</p>',
    unsafe_allow_html=True,
)
