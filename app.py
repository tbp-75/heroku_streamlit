import streamlit as st
import pickle
import pandas as pd
from PIL import Image


st.title("Diabetes Prediction App")
st.write(
    "The data for the following example is originally from the National Institute of Diabetes "
    "and Digestive and Kidney Diseases and contains information on females at least 21 years "
    "old of Pima Indian heritage. This is a sample application and cannot be used as a "
    "substitute for real medical advice."
)
image = Image.open("static/diabetes2.jpeg")
st.image(image, use_column_width=True)
st.write(
    "Please fill in the details of the person under consideration in the left sidebar and "
    "click on the button below!"
)

# SIDE BAR ELEMENTS


Pregnancies = st.sidebar.number_input("Number Of Pregnancies", 0, 5, 0)
Glucose = st.sidebar.slider("Glucose", 40, 200, 80)
BloodPressure = st.sidebar.slider("BloodPressure", 25, 140, 70)
SkinThickness = st.sidebar.slider("SkinThickness", 7, 65, 15)
Insulin = st.sidebar.slider("Insulin", 15, 350, 100)
BMI = st.sidebar.slider("BMI", 10, 60, 20)
DiabetesPedigreeFunction = st.sidebar.slider("DiabetesPedigreeFunction", 0.0, 3.0, 0.2)
Age = st.sidebar.slider("Age", 20, 99, 50)

row = [
    Pregnancies,
    Glucose,
    BloodPressure,
    SkinThickness,
    Insulin,
    BMI,
    DiabetesPedigreeFunction,
    Age,
]


if st.button("Calculate"):

    model = pickle.load(open("models/diabetes.pkl", "rb"))
    scaler = pickle.load(open("models/scaler.pkl", "rb"))

    prediction_values = row
    # Scaled values before sending to model
    prediction_values_scaled = scaler.transform([prediction_values])

    # Make prediction    
    prediction = model.predict_proba(prediction_values_scaled)

    st.markdown("The probability of suffering diabetes is: " + "**{:.1%}**".format(prediction[0,1]))

