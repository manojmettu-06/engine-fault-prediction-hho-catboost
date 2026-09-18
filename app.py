import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Engine Fault Prediction",
    page_icon="⚙️",
    layout="wide"
)


# =========================================================
# TITLE
# =========================================================

st.title("⚙️ Engine Fault Prediction System")

st.write(
    "HHO Optimized CatBoost based Engine Fault Prediction "
    "and Maintenance Recommendation System"
)

st.divider()


# =========================================================
# LOAD DATASET AND MODEL
# =========================================================

DATA_PATH = "dataset/engine_fault_detection_dataset.csv"
MODEL_PATH = "models/baseline_catboost.pkl"

df = pd.read_csv(DATA_PATH)
model = joblib.load(MODEL_PATH)


# =========================================================
# FEATURES AND TARGET
# =========================================================

X = df.drop("Engine_Condition", axis=1)
y = df["Engine_Condition"]


# =========================================================
# TRAIN TEST SPLIT
# =========================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# =========================================================
# MODEL EVALUATION
# =========================================================

y_pred = model.predict(X_test)

y_pred = np.array(y_pred).flatten()

accuracy = accuracy_score(y_test, y_pred)

precision = precision_score(
    y_test,
    y_pred,
    average="weighted",
    zero_division=0
)

recall = recall_score(
    y_test,
    y_pred,
    average="weighted",
    zero_division=0
)

f1 = f1_score(
    y_test,
    y_pred,
    average="weighted",
    zero_division=0
)


# =========================================================
# SIDEBAR - SENSOR INPUT
# =========================================================

st.sidebar.header("🔧 Engine Sensor Parameters")

vibration_amplitude = st.sidebar.number_input(
    "Vibration Amplitude",
    value=float(X["Vibration_Amplitude"].mean())
)

rms_vibration = st.sidebar.number_input(
    "RMS Vibration",
    value=float(X["RMS_Vibration"].mean())
)

vibration_frequency = st.sidebar.number_input(
    "Vibration Frequency",
    value=float(X["Vibration_Frequency"].mean())
)

surface_temperature = st.sidebar.number_input(
    "Surface Temperature",
    value=float(X["Surface_Temperature"].mean())
)

exhaust_temperature = st.sidebar.number_input(
    "Exhaust Temperature",
    value=float(X["Exhaust_Temperature"].mean())
)

acoustic_db = st.sidebar.number_input(
    "Acoustic dB",
    value=float(X["Acoustic_dB"].mean())
)

acoustic_frequency = st.sidebar.number_input(
    "Acoustic Frequency",
    value=float(X["Acoustic_Frequency"].mean())
)

intake_pressure = st.sidebar.number_input(
    "Intake Pressure",
    value=float(X["Intake_Pressure"].mean())
)

exhaust_pressure = st.sidebar.number_input(
    "Exhaust Pressure",
    value=float(X["Exhaust_Pressure"].mean())
)

frequency_band_energy = st.sidebar.number_input(
    "Frequency Band Energy",
    value=float(X["Frequency_Band_Energy"].mean())
)

amplitude_mean = st.sidebar.number_input(
    "Amplitude Mean",
    value=float(X["Amplitude_Mean"].mean())
)


# =========================================================
# PREDICTION
# =========================================================

input_data = pd.DataFrame({
    "Vibration_Amplitude": [vibration_amplitude],
    "RMS_Vibration": [rms_vibration],
    "Vibration_Frequency": [vibration_frequency],
    "Surface_Temperature": [surface_temperature],
    "Exhaust_Temperature": [exhaust_temperature],
    "Acoustic_dB": [acoustic_db],
    "Acoustic_Frequency": [acoustic_frequency],
    "Intake_Pressure": [intake_pressure],
    "Exhaust_Pressure": [exhaust_pressure],
    "Frequency_Band_Energy": [frequency_band_energy],
    "Amplitude_Mean": [amplitude_mean]
})


if st.sidebar.button("🔍 Predict Engine Condition"):

    prediction = model.predict(input_data)

    prediction = int(np.array(prediction).flatten()[0])


    # =====================================================
    # ENGINE CONDITION
    # =====================================================

    st.subheader("🚨 Engine Condition Prediction")

    if prediction == 0:

        condition = "NORMAL"

        st.success(
            f"Predicted Engine Condition: {condition}"
        )

    elif prediction == 1:

        condition = "MINOR FAULT"

        st.warning(
            f"Predicted Engine Condition: {condition}"
        )

    else:

        condition = "CRITICAL FAULT"

        st.error(
            f"Predicted Engine Condition: {condition}"
        )


    # =====================================================
    # MAINTENANCE RECOMMENDATION
    # =====================================================

    st.subheader("🔧 Maintenance Recommendation")

    if prediction == 0:

        st.success(
            "✓ Engine operating normally.\n\n"
            "• Continue routine maintenance.\n\n"
            "• Monitor sensor readings periodically.\n\n"
            "• Perform scheduled inspections."
        )

    elif prediction == 1:

        st.warning(
            "⚠ Minor fault detected.\n\n"
            "• Inspect relevant engine components.\n\n"
            "• Increase monitoring frequency.\n\n"
            "• Schedule maintenance at the earliest suitable opportunity."
        )

    else:

        st.error(
            "🚨 Critical fault detected.\n\n"
            "• Immediate inspection is recommended.\n\n"
            "• Check critical engine components.\n\n"
            "• Avoid continued operation until the fault is assessed."
        )


    st.divider()


# =========================================================
# MODEL PERFORMANCE
# =========================================================

st.subheader("📊 Model Performance")

col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "Accuracy",
    f"{accuracy * 100:.2f}%"
)

col2.metric(
    "Precision",
    f"{precision * 100:.2f}%"
)

col3.metric(
    "Recall",
    f"{recall * 100:.2f}%"
)

col4.metric(
    "F1-Score",
    f"{f1 * 100:.2f}%"
)


st.divider()


# =========================================================
# CONFUSION MATRIX
# =========================================================

st.subheader("🔲 Confusion Matrix")

cm = confusion_matrix(y_test, y_pred)

fig, ax = plt.subplots()

ax.imshow(cm)

ax.set_xlabel("Predicted Label")
ax.set_ylabel("Actual Label")
ax.set_title("Engine Fault Confusion Matrix")

ax.set_xticks([0, 1, 2])
ax.set_yticks([0, 1, 2])

ax.set_xticklabels([
    "Normal",
    "Minor Fault",
    "Critical Fault"
])

ax.set_yticklabels([
    "Normal",
    "Minor Fault",
    "Critical Fault"
])


for i in range(3):

    for j in range(3):

        ax.text(
            j,
            i,
            cm[i, j],
            ha="center",
            va="center"
        )


st.pyplot(fig)

plt.close(fig)


st.divider()


# =========================================================
# CLASSIFICATION REPORT
# =========================================================

st.subheader("📋 Classification Report")

report = classification_report(
    y_test,
    y_pred,
    target_names=[
        "Normal",
        "Minor Fault",
        "Critical Fault"
    ],
    output_dict=True,
    zero_division=0
)

report_df = pd.DataFrame(report).transpose()

st.dataframe(
    report_df.round(4),
    use_container_width=True
)


st.divider()


# =========================================================
# FEATURE IMPORTANCE
# =========================================================

st.subheader("🔍 Feature Importance")

importance = model.get_feature_importance()

feature_importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": importance
})

feature_importance = feature_importance.sort_values(
    by="Importance",
    ascending=False
)

st.dataframe(
    feature_importance.round(4),
    use_container_width=True
)


# =========================================================
# FEATURE IMPORTANCE GRAPH
# =========================================================

fig, ax = plt.subplots()

ax.barh(
    feature_importance["Feature"],
    feature_importance["Importance"]
)

ax.invert_yaxis()

ax.set_xlabel("Importance")
ax.set_ylabel("Sensor Parameter")

ax.set_title(
    "Engine Sensor Feature Importance"
)

st.pyplot(fig)

plt.close(fig)


st.divider()


# =========================================================
# DATASET INFORMATION
# =========================================================

st.subheader("📁 Dataset Information")

col1, col2, col3 = st.columns(3)

col1.metric(
    "Total Samples",
    df.shape[0]
)

col2.metric(
    "Input Features",
    X.shape[1]
)

col3.metric(
    "Output Classes",
    y.nunique()
)


st.write(
    "Dataset contains engine sensor parameters including "
    "vibration, temperature, acoustic and pressure measurements."
)


# =========================================================
# FOOTER
# =========================================================

st.divider()

st.caption(
    "Engine Fault Prediction using HHO and CatBoost"
)