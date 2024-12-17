#deploy
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# App Title
st.title("Healthcare Condition Predictor")
st.write("Predict the likelihood of **Cancer** or **Obesity** based on patient data.")

# Section 1: Upload Dataset
st.header("1. Upload Your Healthcare Dataset")
uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

if uploaded_file:
    st.success("File uploaded successfully!")
    healthcare_data = pd.read_csv(uploaded_file, na_values=["NA", "NaN", "", "?"])

    # Target Variable Simplification
    healthcare_data['Target_Cancer'] = healthcare_data['Medical Condition'].apply(lambda x: 1 if x == 'Cancer' else 0)
    healthcare_data['Target_Obesity'] = healthcare_data['Medical Condition'].apply(lambda x: 1 if x == 'Obesity' else 0)

    # Feature Selection
    # Identify categorical and numerical features
    categorical_features = ['Gender', 'Admission Type', 'Test Results']
    numerical_features = ['Age', 'Billing Amount']

    # Label Encoding for Categorical Features
    label_encoders = {}
    for feature in categorical_features:
        le = LabelEncoder()
        healthcare_data[feature] = le.fit_transform(healthcare_data[feature])
        label_encoders[feature] = le

    # Combine Features for X
    X = healthcare_data[categorical_features + numerical_features]

    # Section 2: Train and Evaluate Model
    st.header("2. Train and Evaluate the Model")
    target_choice = st.selectbox("Select the condition to predict:", ["Cancer", "Obesity"])
    target_column = 'Target_Cancer' if target_choice == 'Cancer' else 'Target_Obesity'
    y = healthcare_data[target_column]

    # Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train Model
    model = LogisticRegression(random_state=42, max_iter=500, class_weight='balanced')
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    conf_matrix = confusion_matrix(y_test, y_pred)

    # Display Results
    st.subheader(f"Model Evaluation for {target_choice}")
    st.write(f"**Accuracy**: {accuracy_score(y_test, y_pred):.4f}")
    st.write(f"**Precision**: {precision_score(y_test, y_pred):.4f}")
    st.write(f"**Recall**: {recall_score(y_test, y_pred):.4f}")
    st.write(f"**F1-Score**: {f1_score(y_test, y_pred):.4f}")

    # Confusion Matrix
    st.write("### Confusion Matrix")
    fig, ax = plt.subplots()
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", 
                xticklabels=['Other', target_choice], yticklabels=['Other', target_choice])
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    st.pyplot(fig)

    # Section 3: Patient Case Study
    st.header("3. Predict for a Custom Patient")
    st.write("Enter the details of a patient to predict the likelihood of the condition.")

    # User Inputs for Patient Features
    patient_data = {}
    for feature in categorical_features:
        options = label_encoders[feature].classes_
        patient_data[feature] = st.selectbox(f"Select {feature}", options)
    for feature in numerical_features:
        patient_data[feature] = st.number_input(f"Enter {feature}", min_value=0.0)

    # Prepare Patient Data
    for feature in categorical_features:
        patient_data[feature] = label_encoders[feature].transform([patient_data[feature]])[0]
    patient_df = pd.DataFrame([patient_data])

    # Prediction
    prediction = model.predict(patient_df)[0]
    probability = model.predict_proba(patient_df)[0, 1]

    # Display Prediction
    st.subheader("Prediction Results")
    st.write(f"**Predicted Class**: {'Positive' if prediction == 1 else 'Negative'}")
    st.write(f"**Predicted Probability**: {probability:.4f}")