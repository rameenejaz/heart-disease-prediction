# Heart Disease Prediction System

A machine learning project that predicts whether a patient has heart disease using only the **4 most discriminating attributes** from the UCI Cleveland Heart Disease dataset.

## Live App
**[Open the app](https://YOUR-APP-NAME.streamlit.app)** (replace with your Streamlit link after deploying)

## Selected Attributes
Chosen with the Chi-Square test (Cramer's V) and Mutual Information. Both methods ranked the same four at the top.

| Attribute | Original Column | Values |
|---|---|---|
| ChestPain | cp | Typical Angina, Atypical Angina, Non-Anginal Pain, Asymptomatic |
| MajorVessels | ca | Zero, One, Two, Three |
| ExerciseAngina | exang | Yes, No |
| Thal | thal | Normal, Fixed Defect, Reversible Defect |

Output: **HeartDisease** (Yes / No)

## Results (Support Vector Classifier)
- Test accuracy: **0.85** (60 test patients)
- Precision / Recall (Disease class): 0.85 / 0.82
- 5-fold cross-validation accuracy: **0.79**

## Files
| File | Description |
|---|---|
| `Heart_Disease_Prediction.ipynb` | Complete notebook: loading, preprocessing, EDA, feature selection, training, evaluation, prediction |
| `Heart_Disease_Report.pdf` | PDF export of the executed notebook |
| `processed_cleveland.csv` | Original UCI Cleveland dataset (303 rows, 14 columns) |
| `heart-sample-data.csv` | Cleaned data with the 4 selected attributes |
| `heart-sample-data-encoded.csv` | Label-encoded version of the above |
| `heart-training-data-encoded.csv` / `heart-testing-data-encoded.csv` | 80/20 train/test split |
| `heart-model-predictions.csv` | Model predictions on the test set |
| `heart_disease_model.pkl` | Trained SVC model |
| `heart_disease_label_encoders.pkl` | Label encoders needed to encode new patient input |
| `app.py` | Streamlit web app (the deployed model) |
| `.streamlit/config.toml` | App theme settings |
| `requirements.txt` | Python package versions |

## How to Run
```
pip install -r requirements.txt
jupyter notebook Heart_Disease_Prediction.ipynb
```
Run all cells top to bottom.

## Run the Web App Locally
```
streamlit run app.py
```

## Use the Saved Model Without Retraining
```python
import pickle, pandas as pd

model = pickle.load(open("heart_disease_model.pkl", "rb"))
encoders = pickle.load(open("heart_disease_label_encoders.pkl", "rb"))

patient = pd.DataFrame({"ChestPain": ["Asymptomatic"], "MajorVessels": ["Two"],
                        "ExerciseAngina": ["Yes"], "Thal": ["Reversible Defect"]})
for col in patient.columns:
    patient[col] = encoders[col].transform(patient[col])

print("Heart Disease:", "Yes" if model.predict(patient)[0] == 1 else "No")
```

## Dataset Source
Detrano, R. et al. Heart Disease dataset (Cleveland). UCI Machine Learning Repository.

*This is an educational project and is not a substitute for medical diagnosis.*
