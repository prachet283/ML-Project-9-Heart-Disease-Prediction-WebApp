import numpy as np
import pickle
import streamlit as st

loaded_model = pickle.load(open("C:/Users/prachet/OneDrive - Vidyalankar Institute of Technology/Desktop/Coding/Machine Learning/ML-Project-9-Heart Disease Prediction/heart_disease_trained_model.sav",'rb'))

def heart_disease_prediction(input_data):

    #changing the input data to numpy
    input_data_as_numpy_array = np.asarray(input_data,dtype=np.float64)

    #reshape the array as we are predicting on 1 instance
    input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

    prediction = loaded_model.predict(input_data_reshaped)
    
    if(prediction[0]==0):
      return 'The person does not have a Heart Disease' 
    else:
      return 'The person have a Heart Disease'


def main():
    
    #giving a title
    st.title('Heart Disease Prediction Web App')
    
    #getting input data from user
        
    col1 , col2 , col3 = st.columns(3)

    with col1:
        age = st.text_input("Age in years")
    with col2:
        sex = st.text_input("Sex (1 = male; 0 = female)")
    with col3:
        chest_pain = st.text_input("Chest Pain type (4 values)")
    with col1:
        resting_bp = st.text_input("Resting Blood Pressure (in mm Hg)")
    with col2:
        serum_cholestoral = st.text_input("Serum Cholestoral in mg/dl")
    with col3:
        fasting_blood_sugar = st.text_input("Fasting Blood Sugar > 120 mg/dl")
    with col1:
        resting_ecg = st.text_input("Resting ECG Results (values 0,1,2)")
    with col2:
        max_heart_achieved = st.text_input("Maximum Heart Rate Achieved")
    with col3:
        exercise_induced_angina = st.text_input("Exercise Induced Angina")
    with col1:
        oldpeak = st.text_input("Oldpeak (ST depression induced by exercise relative to rest)")
    with col2:
        slope_of_peak_exercise = st.text_input("The slope of the peak exercise ST segment")
    with col3:
        number_of_major_vessels = st.text_input("Number of major vessels (0-3) colored by flourosopy")
    with col1:
        thal = st.text_input("Thal: 0 = normal; 1 = fixed defect; 2 = reversable defect")

    # code for prediction
    heart_disease_diagnosis = ''

    #creating a button for Prediction
    if st.button('Heart Disease Test Result'):
        heart_disease_diagnosis=heart_disease_prediction([[age,sex,chest_pain,
                                                resting_bp,serum_cholestoral,fasting_blood_sugar,
                                                resting_ecg,max_heart_achieved,exercise_induced_angina,
                                                oldpeak,slope_of_peak_exercise,
                                                number_of_major_vessels,thal]])
        
    st.success(heart_disease_diagnosis)
    
    
    
if __name__ == '__main__':
    main()
    
    
    
    
    
