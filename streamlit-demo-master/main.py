import streamlit as st 
import numpy as np 
import pandas as pd


import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split

from sklearn.decomposition import PCA
from sklearn.svm import SVR 
from sklearn.metrics import mean_squared_error
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import accuracy_score



dataset_name = st.sidebar.selectbox(
    'Select Dataset',
    ('Supermarket Sales', 'Sales')
)

def selectModel(model):
 if model == 'SVR':
  st.write('You have selected: SVR')
  regressionLinear = SVR (kernel = 'linear')
  regressionLinear.fit (input_train,target_train)
  predicted_outputLinear = regressionLinear.predict(input_test)
  mseLinear = mean_squared_error (predicted_outputLinear, target_test)
  st.write('MSE Value for kernel Linear is',mseLinear)
 elif model == 'KNN':
  st.write ('You have selected: KNN')
  knn_model = KNeighborsRegressor (n_neighbors = 11)
  knn_model.fit(input_train, target_train)
  output_regressor = knn_model.predict(input_test)
  mseKNN = mean_squared_error (output_regressor,target_test)
  st.write('MSE Value for KNN  is with number of neighbors 11',mseKNN)
 elif model == 'Decision Tree':
  st.write('You have selected: Decision Tree')
  dt_model = DecisionTreeRegressor()
  dt_model.fit(input_train, target_train)
  output_dt = dt_model.predict(input_test)
  mseDT = mean_squared_error (output_dt,target_test)
  st.write('MSE Value for Decision Tree is',mseDT)
 else:
  st.write("error detected")

    
if dataset_name == 'Supermarket Sales':
 st.title('Supermarket Sales Dataset')
 st.write('The dataset is about the historical sales of supermarket company which has recorded in 3 different branches for 3 months.The variable input are ')
 st.write('')
 data = pd.read_csv("supermarket_sales.csv")
 data_input = data.drop(columns = ['Invoice ID','Branch','City','Customer type','Gender','Product line','Date','Time','Payment','cogs','gross margin percentage','gross income'])
 data_target = data ['Rating']
 input_train, input_test, target_train, target_test = train_test_split(data_input,data_target,test_size = 0.2)
 st.write('Training Dataset')
 st.write(input_train)
 st.write('Testing Dataset')
 st.write(input_test)
 model = st.selectbox('Select Model',('SVR', 'KNN','Decision Tree'))
 selectModel(model)
 st.title('User Input')

 def user_report():
  unitPrice = st.number_input("Enter unit price(RM)")
  user_report = {
   'unit Price': unitPrice}   
  report_data = pd.DataFrame(user_report, index=[0])
  return report_data

  user_data = user_report()

  regressionLinear = SVR(kernel = 'linear')
  regressionLinear.fit(input_train,target_train)

  st.subheader('Accuracy: ')
  st.write(str(accuracy_score(input_train,regressionLinear.fit(input_test))*100)+'%')

  user_result = regressionLinear.predict(user_data)
  st.subheader('Your weekly sales: RM')
  output = user_result
    
else:
 st.write("tah la nak")
 
    