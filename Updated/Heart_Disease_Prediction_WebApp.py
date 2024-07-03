import json
import pickle
#import numpy as np
import streamlit as st
import pandas as pd


#loading. the saved model
with open("Updated/columns.pkl", 'rb') as f:
    all_columns = pickle.load(f)
with open("Updated/cat_columns.pkl", 'rb') as f:
    cat_columns = pickle.load(f)
with open("Updated/encoder.pkl", 'rb') as f:
    encoder = pickle.load(f)
with open("Updated/encoded_columns.pkl", 'rb') as f:
    encoded_columns = pickle.load(f)
with open("Updated/training_columns.pkl", 'rb') as f:
    training_columns = pickle.load(f)
with open("Updated/scaler.pkl", 'rb') as f:
    scaler = pickle.load(f)
with open("Updated/best_features_xgb.json", 'r') as file:
    best_features_xgb = json.load(file)
with open("Updated/best_features_rfc.json", 'r') as file:
    best_features_rfc = json.load(file)
with open("Updated/best_features_lr.json", 'r') as file:
    best_features_lr = json.load(file)
with open("Updated/heart_disease_trained_xgb_model.sav", 'rb') as f:
    loaded_model_xgb = pickle.load(f)
with open("Updated/heart_disease_trained_rfc_model.sav", 'rb') as f:
    loaded_model_rfc = pickle.load(f)
with open("Updated/heart_disease_trained_lr_model.sav", 'rb') as f:
    loaded_model_lr = pickle.load(f)

def heart_disease_prediction(input_data):

    #loading columns
    columns = all_columns
    
    # Convert the tuple to a DataFrame
    df = pd.DataFrame([input_data], columns=columns)
    
    # Convert the categorical columns to string type
    df[cat_columns] = df[cat_columns].astype('str')
    
    # Encode the categorical columns
    input_data_encoded = encoder.transform(df[cat_columns])
    
    # Create a DataFrame with the encoded features
    input_data_encoded_df = pd.DataFrame(input_data_encoded, columns=encoded_columns)
    
    # Add the remaining non-categorical columns
    input_data_final_encoded = pd.concat([df.drop(cat_columns, axis=1).reset_index(drop=True), input_data_encoded_df], axis=1)
    
    # Standardize the input data
    input_data_scaled = scaler.transform(input_data_final_encoded)
    
    # Create a DataFrame with the standardized features
    input_data_df = pd.DataFrame(input_data_scaled, columns=training_columns)
    
    #loading best features
    df_best_features_xgb = input_data_df[best_features_xgb]
    df_best_features_rfc = input_data_df[best_features_rfc]
    df_best_features_lr = input_data_df[best_features_lr]
    
    #predictions
    prediction1 = loaded_model_xgb.predict(df_best_features_xgb)
    prediction2 = loaded_model_rfc.predict(df_best_features_rfc)
    prediction3 = loaded_model_lr.predict(df_best_features_lr)
    
    return prediction1 , prediction2, prediction3


def main():
    
    #giving a title
    st.title('Heart Disease Prediction Web App')
    
    #getting input data from user
        
    col1 , col2 , col3 = st.columns(3)

    with col1:
        age = st.number_input("Age in years")
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
        resting_bp = st.number_input("Resting Blood Pressure (in mm Hg)")
    with col2:
        serum_cholestoral = st.number_input("Serum Cholestoral in mg/dl")
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
        max_heart_achieved = st.number_input("Maximum Heart Rate Achieved")
    with col3:
        #exercise_induced_angina = st.text_input("Exercise Induced Angina")
        option5 = st.selectbox('Exercise Induced Angina',('Yes', 'No')) 
        exercise_induced_angina = 0 if option5 == 'No' else 1
    with col1:
        oldpeak = st.number_input("Oldpeak (ST depression induced by exercise relative to rest)")
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
    heart_disease_diagnosis_xgb = ''
    heart_disease_diagnosis_rfc = ''
    heart_disease_diagnosis_lr = ''
    heart_disease_diagnosis_xgb,heart_disease_diagnosis_rfc,heart_disease_diagnosis_lr =heart_disease_prediction([age,sex,chest_pain,
                                            resting_bp,serum_cholestoral,fasting_blood_sugar,
                                            resting_ecg,max_heart_achieved,exercise_induced_angina,
                                            oldpeak,slope_of_peak_exercise,
                                            number_of_major_vessels,thal])
    
    
    #creating a button for Prediction
    if st.button("Heart Disease Test Result"):
        if(heart_disease_diagnosis_xgb[0]==0):
            prediction = 'The Person does not have any Heart Disease' 
        else:
            prediction = 'The Person have any Heart Disease'
        st.write(f"Prediction: {prediction}")
    
    if st.checkbox("Show Advanced Options"):
        if st.button("Heart Disease Test Result with XG Boost Classifier"):
            if(heart_disease_diagnosis_xgb[0]==0):
                prediction = 'The Person does not have any Heart Disease' 
            else:
                prediction = 'The Person have any Heart Disease'
            st.write(f"Prediction: {prediction}")
        if st.button("Heart Disease Test Result with Random Forest Classifier"):
            if(heart_disease_diagnosis_rfc[0]==0):
                prediction = 'The Person does not have any Heart Disease' 
            else:
                prediction = 'The Person have any Heart Disease'
            st.write(f"Prediction: {prediction}")
        if st.button("Heart Disease Test Result with Logistics Regression"):
            if(heart_disease_diagnosis_lr[0]==0):
                prediction = 'The Person does not have any Heart Disease' 
            else:
                prediction = 'The Person have any Heart Disease'
            st.write(f"Prediction: {prediction}")
    
if __name__ == '__main__':
    main()
    
    
    
    
    
