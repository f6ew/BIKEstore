#UI script

import streamlit as st
import matplotlib.pyplot as plt
import pandas as dp
import pickle


#--------- Headline ---------
st.title("Forcast your next month!", anchor=None)
st.subheader('Fill the below to know it:')


#--------- taking input data ---------
with st.form(key='my_form'):
    Season = st.select_slider(
     'Season',
    options=[0, 1, 2, 3])
    Month = st.select_slider(
     'Month',
    options=[1, 2, 3, 4, 5, 6, 7,8, 9,10, 11, 12])
    Temp_input = st.number_input(label='Temp')
    aTemp_input = st.number_input(label='aTemp')
    Humidity_input = st.number_input(label='Humidity')
    Weather = st.number_input(label='Weather')
    Wind_input = st.number_input(label='Wind speed')
    button = st.form_submit_button(label='Submit')
   

    def para ():
     
        df_sample = [[float(Season), float(Weather), float(Temp_input), 
        float(aTemp_input),float(Humidity_input), float(Wind_input), float(Month)]]
        
        return df_sample


    dfc = para()
    model = pickle.load(open("model.bin","rb"))
    
if button:
   result = model.predict(dfc)
   st.write("The expected Income :", f'{result}')
        
#--------- present the result --------
#chart_data = pd.DataFrame()
#st.line_chart(chart_data)
