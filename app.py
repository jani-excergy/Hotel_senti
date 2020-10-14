#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 14 01:55:47 2020

@author: janibasha
"""
import streamlit as st
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences



def predict(message):
    
    model=load_model('b_lstm.h5')
    with open('tokenizer.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)
    x_1 = tokenizer.texts_to_sequences([message])
    x_1 = pad_sequences(x_1, maxlen=500)
    predictions = model.predict(x_1)[0][0]
    return predictions

    
st.title("Hotel Reviews Sentiment Classifier ")


hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

import base64

@st.cache(allow_output_mutation=True)
def get_base64_of_bin_file(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

def set_png_as_page_bg(png_file):
    bin_str = get_base64_of_bin_file(png_file)
    page_bg_img = '''
    <style>
    body {
    background-image: url("data:image/png;base64,%s");
    background-size: cover;
    }
    </style>
    ''' % bin_str
    
    st.markdown(page_bg_img, unsafe_allow_html=True)
    return

set_png_as_page_bg('background.png')





message = st.text_area("Please Give Us Your Hotel Experience")
if st.button("Analyze"):
    with st.spinner('Analyzing the text …'):
         prediction=predict(message)
         if prediction > 0.6:
              st.error("Negative review with {:.2f} confidence".format(prediction))
              
        
         elif prediction <0.4:
            st.success("Positive review with {:.2f} confidence".format(1-prediction))
            st.balloons()
        
         else:
             
             st.warning("Not sure! Try to add some more words")


 