import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

# =====================================================
# PAGE CONFIG
# =====================================================

st.set_page_config(
    page_title="Engine Fault Prediction",
    page_icon="⚙️",
    layout="wide"
)

st.title("⚙️ Engine Fault Prediction & Maintenance Decision Support")
st.markdown("### Harris Hawks Optimized CatBoost Framework")

# =====================================================
# LOAD MODEL
# =====================================================

@st.cache_resource
def load_model():
    return joblib.load("models/baseline_catboost.pkl")

model = load_model()

# =====================================================
# SIDEBAR
# =====================================================

st.sidebar.title("Upload Dataset")

uploaded_file = st.sidebar.file_uploader(
    "Upload Engine Sensor CSV",
    type=["csv"]
)

# =====================================================
# MAIN
# =====================================================

if uploaded_file is not None:

    df = pd.read_csv(uploaded_file)

    st.subheader("Uploaded Dataset")

    st.dataframe(df.head())

    if "Engine_Condition" in df.columns:
        X = df.drop("Engine_Condition", axis=1)
    else:
        X = df.copy()

    if st.button("Predict Engine Condition"):

        prediction = model.predict(X)

        df["Predicted_Condition"] = prediction

        mapping = {
            0: "NORMAL",
            1: "WARNING",
            2: "FAULT"
        }

        df["Predicted_Condition"] = df["Predicted_Condition"].map(mapping)

        st.success("Prediction Completed Successfully!")

        st.subheader("Prediction Results")

        st.dataframe(df)

        st.download_button(
            "Download Predictions",
            df.to_csv(index=False),
            "prediction_results.csv",
            "text/csv"
        )

        # =============================================
        # Feature Importance
        # =============================================

        st.subheader("Feature Importance")

        importance = model.get_feature_importance()

        feature_df = pd.DataFrame({
            "Feature": X.columns,
            "Importance": importance
        })

        feature_df = feature_df.sort_values(
            by="Importance",
            ascending=True
        )

        fig, ax = plt.subplots(figsize=(8,6))

        ax.barh(
            feature_df["Feature"],
            feature_df["Importance"]
        )

        ax.set_xlabel("Importance")

        st.pyplot(fig)

        # =============================================
        # Recommendation
        # =============================================

        st.subheader("Maintenance Recommendation")

        result = prediction[0]

        if result == 0:

            st.success("✅ Engine operating normally.")

            st.write("✔ Continue routine maintenance.")
            st.write("✔ Monitor sensor readings periodically.")
            st.write("✔ Perform scheduled inspections.")

        elif result == 1:

            st.warning("⚠ Warning Condition Detected")

            st.write("✔ Inspect vibration system.")
            st.write("✔ Check lubrication.")
            st.write("✔ Monitor engine temperature.")

        else:

            st.error("🚨 Fault Detected")

            st.write("✔ Stop engine immediately.")
            st.write("✔ Perform complete diagnostics.")
            st.write("✔ Replace faulty components.")

else:

    st.info("Please upload an Engine Dataset CSV file.")