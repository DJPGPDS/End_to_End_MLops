import streamlit as st
import pandas as pd
import pickle
import numpy as np
import math

with open('preprocess.pkl', 'rb') as f:
    pre = pickle.load(f)

with open('xg.pkl','rb') as file:
    model = pickle.load(file)

st.header('App for user predictions for Scooter Rental platform')

def user_input_features():
    hr = st.sidebar.slider('Hour of Day', 0, 24, 12)
    weather = st.sidebar.radio('Weather', ['clear', 'cloudy', 'light snow/rain'])
    temperature = st.sidebar.slider("Temp", 30, 140, 60)
    relative_humidity = st.sidebar.slider('Humidity', 0, 100, 30)
    windspeed = st.sidebar.slider('Windspeed', 0, 70, 10)
    year = st.sidebar.slider('Year', 2011, 2012)
    month = st.sidebar.selectbox('Month', ['January', 'February', 'March','April','May','June',
                                           'July','August','September','October','November','December'])
    dayofweek = st.sidebar.selectbox('Day', ['Monday', 'Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday'])

    data = {'hr': hr, 'weather': weather, 'temperature': temperature, 'relative-humidity': relative_humidity, 
            'windspeed': windspeed, 'year': year, 'month': month, 'dayofweek': dayofweek}

    features = pd.DataFrame(data, index=[0])
    return features

df = user_input_features()

# st.write(df)

df_tf = pre.transform(df)

log_prediction = model.predict(df_tf)

result = int(2.71828**log_prediction)


if st.checkbox('Show dataset'):
    st.write('### Scooter rental dataset')
    if st.checkbox('Show transformed dataset'):
        st.write(df_tf)
    elif st.checkbox('Show original dataset'):
        st.write(df)
        

st.subheader('Users Prediction')
st.write(result)
