import streamlit as st
import numpy as np
import pandas as pd
import pickle
import lzma

#Loading sentiment analysis model
with lzma.open("sent_model_svc.xz", "rb") as f:
    loaded_model = pickle.load(f)
    
#Loading vectorizer
with lzma.open("tff_svc.xz", "rb") as f:
    loaded_vectorizer = pickle.load(f)

def main():
  st.markdown("<h1 style='text-align: center; color: White;background-color:#e84343'>DRUG SENTIMENT ANALYSIS</h1>", unsafe_allow_html=True)
  st.markdown("<h3 style='text-align: center; color: White'>Just drop your review and we will do the rest.</h3>", unsafe_allow_html=True)
 
  st.sidebar.header("What is this Project about?")
  st.sidebar.text("This is a web app that is used to predict") 
  st.sidebar.text("user emotions or sentiment from his/her") 
  st.sidebar.text("review about a particular drug to assist") 
  st.sidebar.text("pharmaceutical agents in post marketing")
  st.sidebar.text("Surveillance of drugs")
  
  
  st.sidebar.header("What tools where used to make this?")
  st.sidebar.text("The Model was trained using Support") 
  st.sidebar.text("Vector Machines")
  
  
  
if __name__ =='__main__':
      main() #calling the main method
      
with st.form(key='patient_form'):
   fname = st.text_input("First name: ")
   lname = st.text_input("Last name: ")
   ailment = st.text_input("What's the nature of your sickness: ")
   drugName = st.text_input("What was the name of the drug prescribed to you: ")
   submit_button = st.form_submit_button(label='Submit')
  
reviewl = st.text_input(label="Write a comprehensive review about the drug: ")

    while len(reviewl) < 10:
        st.warning("The review is too short, please write a detailed review. Note: Make use of positive or negative words to help the model predict better. e.g use (I feel bad) instead of (I don't feel good)")
    continue        

inputs = [reviewl] #our inputs
      
if st.button('Predict'): #making and printing our prediction
    result = loaded_model.predict(loaded_vectorizer.transform(inputs))
    if result == 0:
        st.success(f"Thank you for your review {lname}, our algorithm has detected a negative review for {drugName} recommended for {ailment}, please report back again tommorow to let us know how you feel. If feeling persists after 7 days, report physically to where you received the medication.")
    elif result == 1:
        st.success(f"Thank you for your feedback {lname}, our algorithm has detected a positive review for {drugName} recommend for {ailment}. We're happy to see that {drugName} gives positive results for you. Please report daily for the next 2 weeks, this is to ensure that there are no adverse reactions from you on the medication")







