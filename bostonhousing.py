import streamlit as st
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
import seaborn as sns
import shap

# Load the dataset
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/housing/housing.data"
column_names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
boston = pd.read_csv(url, delim_whitespace=True, names=column_names)

# App Title
st.title("House Price Predictor")


# Sidebar for input parameters
st.sidebar.header('Specify Input Parameters')
def user_input_parameters():
    CRIM = st.sidebar.slider('CRIM', float(X.CRIM.min()), float(X.CRIM.max()), float(X.CRIM.mean()))
    ZN = st.sidebar.slider('ZN', float(X.ZN.min()), float(X.ZN.max()), float(X.ZN.mean()))
    INDUS = st.sidebar.slider('INDUS', float(X.INDUS.min()), float(X.INDUS.max()), float(X.INDUS.mean()))
    CHAS = st.sidebar.slider('CHAS', float(X.CHAS.min()), float(X.CHAS.max()), float(X.CHAS.mean()))
    NOX = st.sidebar.slider('NOX', float(X.NOX.min()), float(X.NOX.max()), float(X.NOX.mean()))
    RM = st.sidebar.slider('RM', float(X.RM.min()), float(X.RM.max()), float(X.RM.mean()))
    AGE = st.sidebar.slider('AGE', float(X.AGE.min()), float(X.AGE.max()), float(X.AGE.mean()))
    DIS = st.sidebar.slider('DIS', float(X.DIS.min()), float(X.DIS.max()), float(X.DIS.mean()))
    RAD = st.sidebar.slider('RAD', float(X.RAD.min()), float(X.RAD.max()), float(X.RAD.mean()))
    TAX = st.sidebar.slider('TAX', float(X.TAX.min()), float(X.TAX.max()), float(X.TAX.mean()))
    PTRATIO = st.sidebar.slider('PTRATIO', float(X.PTRATIO.min()), float(X.PTRATIO.max()), float(X.PTRATIO.mean()))
    B = st.sidebar.slider('B', float(X.B.min()), float(X.B.max()), float(X.B.mean()))
    LSTAT = st.sidebar.slider('LSTAT', float(X.LSTAT.min()), float(X.LSTAT.max()), float(X.LSTAT.mean()))

    data = {
        'CRIM': CRIM,
        'ZN': ZN,
        'INDUS': INDUS,
        'CHAS': CHAS,
        'NOX': NOX,
        'RM': RM,
        'AGE': AGE,
        'DIS': DIS,
        'RAD': RAD,
        'TAX': TAX,
        'PTRATIO': PTRATIO,
        'B': B,
        'LSTAT': LSTAT
    }
    features = pd.DataFrame(data, index=[0])
    return features

column_descriptions = {
    'CRIM': 'Per capita crime rate by town.',
    'ZN': 'Proportion of residential land zoned for lots over 25,000 sq. ft.',
    'INDUS': 'Proportion of non-retail business acres per town.',
    'CHAS': 'Charles River dummy variable (1 if tract bounds river; 0 otherwise).',
    'NOX': 'Nitric oxides concentration (parts per 10 million).',
    'RM': 'Average number of rooms per dwelling.',
    'AGE': 'Proportion of owner-occupied units built prior to 1940.',
    'DIS': 'Weighted distances to five Boston employment centers.',
    'RAD': 'Index of accessibility to radial highways.',
    'TAX': 'Full-value property tax rate per $10,000.',
    'PTRATIO': 'Pupil-teacher ratio by town.',
    'B': '1000(Bk - 0.63)^2 where Bk is the proportion of Black residents by town.',
    'LSTAT': 'Percentage of lower status of the population.',
    'MEDV': 'Median value of owner-occupied homes in $1000s.'
}



# Display Column Descriptions
st.subheader('Column Descriptions')
for col, desc in column_descriptions.items():
    st.write(f"**{col}**: {desc}")

# Feature and Target Split
X = boston.drop("MEDV", axis=1)
Y = boston["MEDV"]

df = user_input_parameters()
st.write("---")
st.write(df)

# House Price Prediction
st.subheader('Prediction of MEDV')
model = RandomForestRegressor()
model.fit(X, Y)
prediction = model.predict(df)
st.write(prediction)





# SHAP Values and Feature Importance
st.subheader('Feature Importance based on SHAP values')
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X)

# Standard SHAP summary plot
plt.figure()
shap.summary_plot(shap_values, X)
st.pyplot(plt.gcf())

# SHAP bar plot
plt.figure()
shap.summary_plot(shap_values, X, plot_type="bar")
st.pyplot(plt.gcf())
