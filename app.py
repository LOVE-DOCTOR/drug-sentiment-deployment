import streamlit as st
import numpy as np
import string
import pandas as pd
import pickle
import lzma
import datetime

#Loading sentiment analysis model
with lzma.open("sent_model_svc.xz", "rb") as f:
    loaded_model = pickle.load(f)

#Loading vectorizer
with lzma.open("tff_svc.xz", "rb") as f:
    loaded_vectorizer = pickle.load(f)



def main():
  st.markdown("<h1 style='text-align: center; color: White;background-color:#e84343'>DRUG SENTIMENT ANALYSIS</h1>", unsafe_allow_html=True)
  st.markdown("<h3 style='text-align: center; color: White'>Just drop your review and we will do the rest.</h3>", unsafe_allow_html=True)
 
  #st.sidebar.header("What is this Project about?")
  #st.sidebar.text("This is a web app that is used to predict") 
  #st.sidebar.text("user emotions or sentiment from his/her") 
  #st.sidebar.text("review about a particular drug to assist") 
  #st.sidebar.text("doctors in clinical trials")
  
  #st.sidebar.header("What tools where used to make this?")
  #st.sidebar.text("The Model was trained using Support") 
  #st.sidebar.text("Vector Machines with a dataset from") 
  #st.sidebar.text("UCI Machine Learning Repository")
  
  
  
if __name__ =='__main__':
      main() #calling the main method
      
with st.form(key='patient_form'):
   date = st.date_input("Select today's date: ")
   fname = st.text_input("First name: ")
   lname = st.text_input("Last name: ")
   age = st.number_input("Your age: ", step=1)
   ailment = st.text_input("What's the nature of your sickness: ")
   drugName = st.text_input("What was the name of the drug prescribed to you: ")
   submit_button = st.form_submit_button(label='Submit')

if submit_button:
    st.success("Be rest assured! Your information is safe with us.")  
    
reviewl = st.text_input(label="Write a review about the drug here: You can help the model to better predict the mood of your text if you avoid using negation. e.g Use (my health has worsened) instead of (my health has not improved). Now you can write your own review.")

inputs = [reviewl] #our inputs
    
      
if st.button('Predict'): #making and printing our prediction
    result = loaded_model.predict(loaded_vectorizer.transform(inputs))
    if result == 0:
        st.success(f"Thank you for your review {lname}, our algorithm has detected a negative review for {drugName} recommended for {ailment}, please report back again tomorrow to let us know how you feel. If feeling persists, after 7 days, report physically to where you received the medication.")
        
        st.header(f"SERIOUS NEGATIVE REACTIONS TO {drugName}?")
        with st.expander("If yes, please set an appointment here: "):
            with st.form(key='Appointment_form'):
                appointment_date = st.date_input("Pick a date: ")
                appointment_time = st.time_input("Pick a time: ")
                appointment_button = st.form_submit_button(label='Submit')
                if appointment_button:
                    st.success("You have successfully set an appoinment, an email will be sent to you as a reminder a day to the appointment")

    elif result == 1:
        st.success(f"Thank you for the positive feeback {lname}, our algorithm has detected a positive review for {drugName} recommended for {ailment}. We're happy to see that {drugName} gives positive results for you. Please report daily for the next 2 weeks, this is to ensure that there are no adverse reactions from you due to the medication.")





