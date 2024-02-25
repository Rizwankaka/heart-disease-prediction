import streamlit as st
import numpy as np
from joblib import load

# Load the model
model = load('model.joblib')

def preprocess_input(sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal):
    # Initialize all features with zeros or appropriate default values
    features = np.zeros(18)

    # Encode sex: Assuming 1st position for male, 2nd for female (0-indexed)
    if sex == 'Male':
        features[0] = 1
    else:
        features[1] = 1

    # Chest Pain Type (cp): Assuming positions 2-5 for one-hot encoding
    if 1 <= cp <= 4:
        features[2 + (cp - 1)] = 1

    # Numerical features directly assigned
    features[6] = trestbps
    features[7] = chol
    features[8] = thalach
    features[9] = oldpeak
    features[10] = ca

    # Fasting Blood Sugar (fbs): Assuming position 11, binary encoding
    features[11] = 1 if fbs == 'Yes' else 0

    # Resting ECG (restecg): Assuming positions 12-14 for one-hot encoding
    if 0 <= restecg <= 2:
        features[12 + restecg] = 1

    # Exercise Induced Angina (exang): Assuming position 15, binary encoding
    features[15] = 1 if exang == 'Yes' else 0

    # Slope: Assuming positions 16-17 for one-hot encoding (assuming 2 categories for simplicity)
    if 0 <= slope <= 1:
        features[16 + slope] = 1

    # Adjust for thal: This feature is not directly included in the example to match the 18 feature constraint
    # If thal is crucial, adjust the feature positions and encoding strategy accordingly

    return features.reshape(1, -18)

def main():
    st.title('Heart Disease Prediction')

    # Sidebar inputs
    sex = st.sidebar.selectbox('Sex', ['Male', 'Female'])
    cp = st.sidebar.selectbox('Chest Pain Type', [1, 2, 3, 4])
    trestbps = st.sidebar.slider('Resting Blood Pressure', 94, 200, 120)
    chol = st.sidebar.slider('Cholesterol', 126, 564, 240)
    fbs = st.sidebar.selectbox('Fasting Blood Sugar > 120 mg/dl', ['Yes', 'No'])
    restecg = st.sidebar.selectbox('Resting Electrocardiographic Results', [0, 1, 2])
    thalach = st.sidebar.slider('Max Heart Rate Achieved', 71, 202, 150)
    exang = st.sidebar.selectbox('Exercise Induced Angina', ['Yes', 'No'])
    oldpeak = st.sidebar.slider('ST depression induced by exercise relative to rest', 0.0, 6.2, 1.0)
    slope = st.sidebar.selectbox('The slope of the peak exercise ST segment', [0, 1])
    ca = st.sidebar.slider('Number of major vessels colored by flourosopy', 0, 4, 0)
    thal = st.sidebar.selectbox('Thalassemia', [0, 1, 2, 3])  # Included for completeness, adjust usage as needed

    if st.button('Predict'):
        features = preprocess_input(sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal)
        prediction = model.predict(features)
        st.write('Prediction (0: No Heart Disease, 1: Heart Disease):', prediction[0])

if __name__ == '__main__':
    main()
