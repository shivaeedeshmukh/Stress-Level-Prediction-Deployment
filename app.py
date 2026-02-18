import streamlit as st
import numpy as np
import pickle


best_rf = pickle.load(open("best_rf.pkl","rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

st.title("Stress Level Prediction App")

st.sidebar.header("Enter Details")

gender = st.sidebar.selectbox("Gender",["Male","Female"])
age = st.sidebar.number_input("Age", 18, 80)
sleep_duration = st.sidebar.number_input("Sleep Duration")
quality_sleep = st.sidebar.slider("Quality of sleep (1-10)" ,1, 10)
physical_activity = st.sidebar.number_input("Physical Activity Level")
bmi = st.sidebar.number_input("BMI Category")
heart_rate = st.sidebar.number_input("Heart Rate")
daily_steps = st.sidebar.number_input("Daily Steps")
systolic = st.sidebar.number_input("Systolic BP")
diastolic = st.sidebar.number_input("Diastolic BP")


sleep_disorder = st.sidebar.selectbox("Sleep Disorder", ["None","Sleep Apnea","Insomnia"])
occupation = st.sidebar.selectbox("Occupation", [ "Doctor", "Engineer", "Lawyer", "Manager", "Nurse",
    "Sales Representative", "Salesperson",
    "Scientist", "Software Engineer", "Teacher"])

gender_val = 1 if gender == "Male" else 0


sleep_none = 1 if sleep_disorder == "None" else 0
sleep_apnea = 1 if sleep_disorder == "Sleep Apnea" else 0

occupation_list = [
    "Doctor", "Engineer", "Lawyer", "Manager", "Nurse",
    "Sales Representative", "Salesperson",
    "Scientist", "Software Engineer", "Teacher"
]


occupation_encoded = [1 if occupation == occ else 0 for occ in occupation_list]

input_data = [
    gender_val, age, sleep_duration, quality_sleep,
    physical_activity, bmi, heart_rate, daily_steps,
    sleep_none, sleep_apnea, 
    *occupation_encoded,
    systolic, diastolic
]

input_array = np.array([input_data])
input_scaled = scaler.transform(input_array)



if st.button("Predict Stress Level"):
    prediction = best_rf.predict(input_scaled)
    st.success(f"Predicted Stress Level: {prediction[0]}")