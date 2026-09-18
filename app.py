import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="Engine Fault Prediction",
    page_icon="⚙️",
    layout="wide"
)

st.title("⚙️ Engine Fault Prediction System")

st.write(
    "Upload an engine sensor CSV file to predict engine condition "
    "using CatBoost."
)

st.divider()


# =========================================================
# LOAD MODEL
# =========================================================

MODEL_PATH = "models/baseline_catboost.pkl"

try:
    model = joblib.load(MODEL_PATH)
except Exception as e:
    st.error(f"Unable to load model: {e}")
    st.stop()


# =========================================================
# REQUIRED FEATURES
# =========================================================

FEATURES = [
    "Vibration_Amplitude",
    "RMS_Vibration",
    "Vibration_Frequency",
    "Surface_Temperature",
    "Exhaust_Temperature",
    "Acoustic_dB",
    "Acoustic_Frequency",
    "Intake_Pressure",
    "Exhaust_Pressure",
    "Frequency_Band_Energy",
    "Amplitude_Mean"
]


# =========================================================
# CSV UPLOAD
# =========================================================

st.header("📁 Upload Engine Dataset")

uploaded_file = st.file_uploader(
    "Upload CSV file",
    type=["csv"]
)


if uploaded_file is None:

    st.info(
        "Please upload a CSV file containing the required engine "
        "sensor parameters."
    )

    st.write("### Required columns")

    for feature in FEATURES:
        st.write(f"• {feature}")

    st.write("Optional column for evaluation:")

    st.write("• Engine_Condition")

    st.stop()


# =========================================================
# READ CSV
# =========================================================

try:

    data = pd.read_csv(uploaded_file)

except Exception as e:

    st.error(f"Unable to read CSV file: {e}")
    st.stop()


st.success("CSV file uploaded successfully!")

st.subheader("📋 Uploaded Dataset")

st.write(
    f"Rows: **{data.shape[0]}** | "
    f"Columns: **{data.shape[1]}**"
)

st.dataframe(
    data.head(10),
    width="stretch"
)


# =========================================================
# CHECK FEATURES
# =========================================================

missing_features = [
    feature for feature in FEATURES
    if feature not in data.columns
]


if missing_features:

    st.error("The uploaded CSV is missing required columns:")

    for feature in missing_features:
        st.write(f"❌ {feature}")

    st.stop()


# =========================================================
# PREPARE INPUT DATA
# =========================================================

X = data[FEATURES].copy()


# =========================================================
# HANDLE MISSING VALUES
# =========================================================

if X.isnull().sum().sum() > 0:

    st.warning(
        "Missing values detected. They will be replaced "
        "using column mean values."
    )

    X = X.fillna(X.mean())


# =========================================================
# PREDICTION
# =========================================================

try:

    predictions = model.predict(X)

    predictions = np.array(predictions).flatten()

except Exception as e:

    st.error(f"Prediction failed: {e}")
    st.stop()


# =========================================================
# PREDICTION RESULTS
# =========================================================

st.divider()

st.header("🚨 Prediction Results")


condition_names = {
    0: "NORMAL",
    1: "MINOR FAULT",
    2: "CRITICAL FAULT"
}


result_data = data.copy()

result_data["Predicted_Engine_Condition"] = [
    condition_names.get(int(p), str(p))
    for p in predictions
]


# =========================================================
# DISPLAY CONDITION COUNTS
# =========================================================

prediction_counts = pd.Series(
    [
        condition_names.get(int(p), str(p))
        for p in predictions
    ]
).value_counts()


col1, col2, col3 = st.columns(3)


col1.metric(
    "Normal",
    prediction_counts.get("NORMAL", 0)
)

col2.metric(
    "Minor Fault",
    prediction_counts.get("MINOR FAULT", 0)
)

col3.metric(
    "Critical Fault",
    prediction_counts.get("CRITICAL FAULT", 0)
)


st.subheader("Prediction Table")

st.dataframe(
    result_data,
    width="stretch"
)


# =========================================================
# EVALUATION
# =========================================================

if "Engine_Condition" in data.columns:

    st.divider()

    st.header("📊 Model Performance")

    y_true = data["Engine_Condition"]

    accuracy = accuracy_score(
        y_true,
        predictions
    )

    precision = precision_score(
        y_true,
        predictions,
        average="weighted",
        zero_division=0
    )

    recall = recall_score(
        y_true,
        predictions,
        average="weighted",
        zero_division=0
    )

    f1 = f1_score(
        y_true,
        predictions,
        average="weighted",
        zero_division=0
    )


    # -----------------------------------------------------
    # METRICS
    # -----------------------------------------------------

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


    # -----------------------------------------------------
    # CONFUSION MATRIX
    # -----------------------------------------------------

    st.subheader("🔲 Confusion Matrix")

    cm = confusion_matrix(
        y_true,
        predictions,
        labels=[0, 1, 2]
    )

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


    # -----------------------------------------------------
    # CLASSIFICATION REPORT
    # -----------------------------------------------------

    st.subheader("📋 Classification Report")

    report = classification_report(
        y_true,
        predictions,
        labels=[0, 1, 2],
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
        width="stretch"
    )


else:

    st.info(
        "Engine_Condition column was not found. "
        "Predictions are available, but evaluation metrics "
        "cannot be calculated without actual labels."
    )


# =========================================================
# FEATURE IMPORTANCE
# =========================================================

st.divider()

st.header("🔍 Feature Importance")

importance = model.get_feature_importance()

feature_importance = pd.DataFrame({
    "Feature": FEATURES,
    "Importance": importance
})

feature_importance = feature_importance.sort_values(
    by="Importance",
    ascending=False
)


st.dataframe(
    feature_importance.round(4),
    width="stretch"
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


# =========================================================
# MAINTENANCE RECOMMENDATION
# =========================================================

st.divider()

st.header("🔧 Maintenance Recommendation")


critical_count = prediction_counts.get(
    "CRITICAL FAULT",
    0
)

minor_count = prediction_counts.get(
    "MINOR FAULT",
    0
)

normal_count = prediction_counts.get(
    "NORMAL",
    0
)


if critical_count > 0:

    st.error(
        f"🚨 {critical_count} critical fault prediction(s) detected."
    )

    st.write(
        "• Immediate inspection is recommended."
    )

    st.write(
        "• Check critical engine components."
    )

    st.write(
        "• Avoid continued operation until the fault is assessed."
    )


elif minor_count > 0:

    st.warning(
        f"⚠ {minor_count} minor fault prediction(s) detected."
    )

    st.write(
        "• Inspect relevant engine components."
    )

    st.write(
        "• Increase monitoring frequency."
    )

    st.write(
        "• Schedule maintenance at the earliest suitable opportunity."
    )


else:

    st.success(
        f"✓ All {normal_count} prediction(s) indicate NORMAL operation."
    )

    st.write(
        "• Continue routine maintenance."
    )

    st.write(
        "• Monitor sensor readings periodically."
    )

    st.write(
        "• Perform scheduled inspections."
    )


# =========================================================
# FOOTER
# =========================================================

st.divider()

st.caption(
    "Engine Fault Prediction using CatBoost | "
    "Predictive Maintenance Support System"
)