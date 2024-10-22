import pandas as pd
import numpy as np
import streamlit as st
import pickle

def data():
    df = pd.read_csv("Student_Performance.csv")
    df["Extracurricular Activities"]=df["Extracurricular Activities"].map({"Yes":2,"No":1})
    return df   

def add_sidebar():
    df=data()
    slider_label=[("Hours Studied","Hours Studied"),
                  ("Previous Scores","Previous Scores"),
                  ("Extracurricular Activities","Extracurricular Activities"),
                  ("Sleep Hours","Sleep Hours"),
                  ("Sample Question Papers Practiced","Sample Question Papers Practiced")]
    
    input_dict={}
    for label,key in slider_label:
        input_dict[key]=st.sidebar.slider(
            label,
            min_value=0,
            max_value=int(df[key].max())
            )
    return input_dict

def add_prediction(input_data):
    df=data()
    with open("model.pkl","rb") as model_in:
        classifier=pickle.load(model_in)
        
    input_data_normal=list(input_data.values())
    input_data_np=np.array(input_data_normal).reshape(1,-1)
    
    prediction=classifier.predict(input_data_np).astype(int)
    
    st.subheader("Prediction Result")
    prediction[0][0]
    if prediction[0]<=33: 
        st.write("Poor performance")
    elif 34<prediction[0]<=60:
        st.write("Average Performance")
    elif 61<prediction[0]<70:
        st.write("Good performance")
    else:
        st.write("Excellent Performance")

def main():
    st.set_page_config(
        page_title="Student Performance Index prediction app",
        layout="wide",
        initial_sidebar_state="expanded")
    
    input_data=add_sidebar()
    
    with st.container():
        st.header("Prediction app")
        st.write("This app's prediction is totally based on the student data provided and is only used for predicting the performance index only")
        
    if st.button("predict"):
        add_prediction(input_data)

if __name__=="__main__":
    main()