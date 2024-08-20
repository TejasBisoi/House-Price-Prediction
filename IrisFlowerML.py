import streamlit as st
import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier

st.write(
    #Iris Flower Prediction App
    ##This app predicts the iris flower type!

)
st.sidebar.header("User Input Parameter")

def user_input_parameters():
        sepal_length=st.sidebar.slider('Sepal Length', 4.3, 7.9, 5.4),  # minimum , maximum, default values
        sepal_width=st.sidebar.slider('Sepal Width', 2.0, 4.4, 3.4),
        petal_length=st.sidebar.slider('Petal Length', 1.0, 6.9, 1.3),
        petal_width=st.sidebar.slider('Petal Width', 0.1, 2.5, 0.2)
    
        data={
        'sepal_length':sepal_length,
        'sepal_width':sepal_width, 
        'petal_length':petal_length,
        'petal_width':petal_width}

    
        features=pd.DataFrame(data, index=[0])

        return features

df=user_input_parameters()
st.write(df)

st.subheader("User Input Parameters")
iris=datasets.load_iris()

X=iris.data
Y=iris.target
clf=RandomForestClassifier()
clf.fit(X,Y)

prediction=clf.predict(df)
prediction_proba=clf.predict_proba(df)

st.subheader("Class Labels and Their Corresponding Index Number")
st.write(iris.target_names)

st.subheader('Prediction Probability')
st.write(prediction_proba)

