"""
Heart Disease Prediction - Streamlit App
----------------------------------------
This app loads the trained SVC model (svc_trained_model.pkl) and the trained
scaler (scaler.pkl) that were saved by the notebook in Step 6.4, collects the
4 best patient attributes selected in Step 4 (cp, thalach, ca, thal), and
shows the prediction.

This is a learning project. It is NOT a medical diagnosis tool.
"""

import os
import pickle

import pandas as pd
import streamlit as st

# ---------------------------------------------------------------- Load model and scaler
# The .pkl files must be in the same folder as this app.py file
folder = os.path.dirname(os.path.abspath(__file__))
model = pickle.load(open(os.path.join(folder, "svc_trained_model.pkl"), "rb"))
scaler = pickle.load(open(os.path.join(folder, "scaler.pkl"), "rb"))

# ---------------------------------------------------------------- Page title
st.set_page_config(page_title="Heart Disease Prediction", page_icon="❤️")
st.title("Heart Disease Prediction System")
st.write("Enter the patient's details below and click **Predict**. "
         "This is a learning project and is **not** a medical diagnosis tool.")
st.caption("The model uses the 4 most informative attributes of the Cleveland dataset "
           "(selected from the original 13 in Step 4 of the notebook).")

# ---------------------------------------------------------------- Input fields
# Each choice shows readable text, but the number after it is the code used in the dataset
cp_choice = st.selectbox("Chest pain type", ["Typical angina", "Atypical angina", "Non-anginal pain", "Asymptomatic"], index=3)
cp = {"Typical angina": 1, "Atypical angina": 2, "Non-anginal pain": 3, "Asymptomatic": 4}[cp_choice]

thalach = st.number_input("Maximum heart rate achieved", min_value=40, max_value=250, value=120)

ca = st.selectbox("Number of major vessels coloured by fluoroscopy", [0, 1, 2, 3])

thal_choice = st.selectbox("Thal", ["Normal", "Fixed defect", "Reversible defect"], index=2)
thal = {"Normal": 3, "Fixed defect": 6, "Reversible defect": 7}[thal_choice]

# ---------------------------------------------------------------- Predict
if st.button("Predict"):
    # Step 8.2: build the feature vector (same columns, same order as the notebook)
    user_input = pd.DataFrame({
        "cp": [cp], "thalach": [thalach], "ca": [ca], "thal": [thal],
    })[list(scaler.feature_names_in_)]

    # Step 8.3: scale the feature vector with the saved scaler
    unseen_data_features = pd.DataFrame(scaler.transform(user_input), columns=user_input.columns)

    # Step 8.5.1: predict
    predicted_heart_disease = model.predict(unseen_data_features)[0]

    if predicted_heart_disease == 1:
        st.error("Prediction: HEART DISEASE PRESENT")
    else:
        st.success("Prediction: NO HEART DISEASE")
