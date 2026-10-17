import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import joblib


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Hospital Readmission Prediction",
    page_icon="🏥",
    layout="wide"
)


# =========================================================
# LOAD MODEL AND PREPROCESSING FILES
# =========================================================

MODEL_PATH = "data/hospital_readmission_model.keras"
SCALER_PATH = "data/scaler.pkl"
FEATURE_PATH = "data/feature_columns.pkl"

@st.cache_resource
def load_model_and_preprocessing():

    model = tf.keras.models.load_model(MODEL_PATH)

    scaler = joblib.load(SCALER_PATH)

    feature_columns = joblib.load(FEATURE_PATH)

    return model, scaler, feature_columns


model, scaler, feature_columns = load_model_and_preprocessing()


# =========================================================
# HEADER
# =========================================================

st.title("🏥 Hospital Readmission Prediction System")

st.markdown(
    """
    ### Neural Network Based Prediction

    This system uses an **Artificial Neural Network (ANN)** to predict
    whether a patient is likely to be readmitted based on patient
    and hospital-related information.
    """
)

st.divider()


# =========================================================
# PATIENT INFORMATION
# =========================================================

st.subheader("👤 Patient Information")

col1, col2, col3 = st.columns(3)

with col1:
    patient_name = st.text_input(
        "Patient Name",
        placeholder="Enter patient name"
    )

with col2:
    age = st.number_input(
        "Age",
        min_value=1,
        max_value=120,
        value=50
    )

with col3:
    gender = st.selectbox(
        "Gender",
        ["Male", "Female"]
    )


# =========================================================
# MEDICAL INFORMATION
# =========================================================

st.subheader("🩺 Medical Information")

col1, col2, col3 = st.columns(3)

with col1:
    time_in_hospital = st.number_input(
        "Time in Hospital (days)",
        min_value=1,
        max_value=30,
        value=3
    )

with col2:
    num_lab_procedures = st.number_input(
        "Number of Lab Procedures",
        min_value=0,
        max_value=150,
        value=40
    )

with col3:
    num_procedures = st.number_input(
        "Number of Procedures",
        min_value=0,
        max_value=50,
        value=1
    )


col1, col2, col3 = st.columns(3)

with col1:
    num_medications = st.number_input(
        "Number of Medications",
        min_value=0,
        max_value=100,
        value=10
    )

with col2:
    number_outpatient = st.number_input(
        "Outpatient Visits",
        min_value=0,
        max_value=100,
        value=0
    )

with col3:
    number_emergency = st.number_input(
        "Emergency Visits",
        min_value=0,
        max_value=100,
        value=0
    )


col1, col2, col3 = st.columns(3)

with col1:
    number_inpatient = st.number_input(
        "Previous Inpatient Visits",
        min_value=0,
        max_value=100,
        value=0
    )

with col2:
    number_diagnoses = st.number_input(
        "Number of Diagnoses",
        min_value=0,
        max_value=20,
        value=5
    )

with col3:
    insulin = st.selectbox(
        "Insulin",
        ["No", "Steady", "Up", "Down"]
    )


st.divider()


# =========================================================
# PREDICTION
# =========================================================

if st.button("🔍 Predict Readmission", use_container_width=True):

    if patient_name.strip() == "":
        st.warning("Please enter the patient's name.")

    else:

        # -------------------------------------------------
        # Load original training dataset
        # -------------------------------------------------

        df = pd.read_csv("Train.csv")

        # Remove target column
        X = df.drop("readmitted", axis=1)

        # -------------------------------------------------
        # Create a patient record
        # -------------------------------------------------

        patient = X.iloc[[0]].copy()

        # Numerical values
        if "age" in patient.columns:
            patient["age"] = age

        if "time_in_hospital" in patient.columns:
            patient["time_in_hospital"] = time_in_hospital

        if "num_lab_procedures" in patient.columns:
            patient["num_lab_procedures"] = num_lab_procedures

        if "num_procedures" in patient.columns:
            patient["num_procedures"] = num_procedures

        if "num_medications" in patient.columns:
            patient["num_medications"] = num_medications

        if "number_outpatient" in patient.columns:
            patient["number_outpatient"] = number_outpatient

        if "number_emergency" in patient.columns:
            patient["number_emergency"] = number_emergency

        if "number_inpatient" in patient.columns:
            patient["number_inpatient"] = number_inpatient

        if "number_diagnoses" in patient.columns:
            patient["number_diagnoses"] = number_diagnoses

        # -------------------------------------------------
        # Gender
        # -------------------------------------------------

        if "gender" in patient.columns:
            patient["gender"] = gender

        # -------------------------------------------------
        # Insulin
        # -------------------------------------------------

        if "insulin" in patient.columns:
            patient["insulin"] = insulin

        # -------------------------------------------------
        # One-hot encoding
        # -------------------------------------------------

        patient_encoded = pd.get_dummies(patient)

        # Make columns exactly the same as training data
        patient_encoded = patient_encoded.reindex(
            columns=feature_columns,
            fill_value=0
        )

        # -------------------------------------------------
        # Scaling
        # -------------------------------------------------

        patient_scaled = scaler.transform(patient_encoded)

        # -------------------------------------------------
        # Prediction
        # -------------------------------------------------

        prediction = model.predict(patient_scaled, verbose=0)

        probability = float(prediction[0][0])

        # -------------------------------------------------
        # Display result
        # -------------------------------------------------

        st.divider()

        st.subheader("📊 Prediction Result")

        st.write(f"**Patient:** {patient_name}")

        st.write(f"**Age:** {age}")

        st.write(
            f"**Readmission Probability:** {probability * 100:.2f}%"
        )

        if probability >= 0.5:

            st.error(
                "⚠️ Prediction: Patient is likely to be readmitted"
            )

        else:

            st.success(
                "✅ Prediction: Patient is unlikely to be readmitted"
            )

        st.info(
            "This prediction is generated by the trained Artificial "
            "Neural Network model and is intended for academic "
            "demonstration purposes."
        )


# =========================================================
# FOOTER
# =========================================================

st.divider()

st.caption(
    "Hospital Readmission Prediction using Artificial Neural Network | "
    "Academic Project"
)