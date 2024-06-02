import numpy as np
import pickle
import streamlit as st

loaded_model = pickle.load(open("heart_disease_trained_model.sav",'rb'))

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
        #sex = st.text_input("Sex (1 = male; 0 = female)")
        option1 = st.selectbox('Gender',('Male', 'Female')) 
        sex = 0 if option1 == 'Female' else 1
    with col3:
        #chest_pain = st.text_input("Chest Pain type")
        option2 = st.selectbox('Chest Pain type',('0','1','2','3'))
        if option2 == '0':
            chest_pain = 0
        elif option2 == '1':
            chest_pain = 1
        elif option2 == '2':
            chest_pain = 2
        else:
            chest_pain = 3
    with col1:
        resting_bp = st.text_input("Resting Blood Pressure (in mm Hg)")
    with col2:
        serum_cholestoral = st.text_input("Serum Cholestoral in mg/dl")
    with col3:
        #fasting_blood_sugar = st.text_input("Fasting Blood Sugar > 120 mg/dl")
        option3 = st.selectbox('Fasting Blood Sugar',('True', 'False')) 
        fasting_blood_sugar = 0 if option3 == 'False' else 1
    with col1:
        #resting_ecg = st.text_input("Resting ECG Results (values 0,1,2)")
        option4 = st.selectbox('Resting ECG Results',('0','1','2','3'))
        if option4 == '0':
            resting_ecg = 0
        elif option4 == '1':
            resting_ecg = 1
        elif option4 == '2':
            resting_ecg = 2
    with col2:
        max_heart_achieved = st.text_input("Maximum Heart Rate Achieved")
    with col3:
        #exercise_induced_angina = st.text_input("Exercise Induced Angina")
        option5 = st.selectbox('Exercise Induced Angina',('Yes', 'No')) 
        exercise_induced_angina = 0 if option5 == 'No' else 1
    with col1:
        oldpeak = st.text_input("Oldpeak (ST depression induced by exercise relative to rest)")
    with col2:
        #slope_of_peak_exercise = st.text_input("The slope of the peak exercise ST segment")
        option6 = st.selectbox('The slope of the peak exercise ST segment',('0','1','2'))
        if option6 == '0':
            slope_of_peak_exercise = 0
        elif option6 == '1':
            slope_of_peak_exercise = 1
        elif option6 == '2':
            slope_of_peak_exercise = 2
    with col3:
        #number_of_major_vessels = st.text_input("Number of major vessels (0-4) colored by flourosopy")
        option7 = st.selectbox('The slope of the peak exercise ST segment',('0','1','2','3','4'))
        if option7 == '0':
            number_of_major_vessels = 0
        elif option7 == '1':
            number_of_major_vessels = 1
        elif option7 == '2':
            number_of_major_vessels = 2
        elif option7 == '3':
            number_of_major_vessels = 3
        else:
            number_of_major_vessels = 4

    with col1:
        #thal = st.text_input("Thal: 0 = normal; 1 = fixed defect; 2 = reversable defect")
        option7 = st.selectbox('Thal',('None','Normal','Fixed defect','Reversable defect'))
        if option7 == 'None':
            thal = 0
        elif option7 == 'Normal':
            thal = 1
        elif option7 == 'Fixed defect':
            thal = 2
        elif option7 == 'Reversable defect':
            thal = 3

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
    
    
    
    
    
