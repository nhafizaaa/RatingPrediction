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

st.markdown(
    """
    <style>
    .reportview-container {
        background-color: #00000
    }
   .sidebar .sidebar-content {
        background-color: #CE9ABB
    }
    </style>
    """,
    unsafe_allow_html=True
)

dataset_name = st.sidebar.selectbox(
    'Select Dataset',
    ('Supermarket Sales', 'Sales','Retail Sales','Yearly Sales Prediction')
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

#PYJAAAA
if dataset_name == 'Supermarket Sales':
    st.title('Supermarket Sales Dataset')
    st.write('The dataset is about the historical sales of supermarket company which has recorded in 3 different branches for 3 months.The variable input are ')
    st.write('')
    data = pd.read_csv("supermarket_sales.csv")
    data_input = data.drop(columns = ['Invoice ID','Branch','City','Customer type','Gender','Product line','Date','Time','Payment','cogs','gross margin percentage','gross income','Rating','Quantity','Tax 5%', 'Total'])
    data_target = data ['Rating']
    input_train, input_test, target_train, target_test = train_test_split(data_input,data_target,test_size = 0.2)
    st.write('Training Dataset')
    st.write(input_train)
    st.write('Testing Dataset')
    st.write(input_test)
    model = st.selectbox('Select Model',('SVR', 'KNN','Decision Tree'))
    selectModel(model)

    st.title('User Input')

    def user_report1():
        Unitprice = st.slider('Unit price (RM)',1,100,1)
      
        user_report1 = {
            'Unit price': Unitprice, 
         }

        report_data1 = pd.DataFrame(user_report1, index=[0])
        return report_data1

    user_data = user_report1()

    regressionLinear = SVR(kernel = 'linear')
    regressionLinear.fit(input_train,target_train)

    user_result = regressionLinear.predict(user_data)
    st.subheader('Rating: ')
    output = user_result

    st.write(output)

#YENNN
elif dataset_name == 'Retail Sales':
    st.title('Retail Sales')
    st.write('The dataset is about sales bla bla')
    st.write('')
    data = pd.read_csv("retail_sales.csv")
    data_input = data.drop(columns = ['No', 'Total Orders','Gross Sales','Discounts', 'Returns', 'Net Sales', 'Shipping', 'Total Sales'])
    data_target = data ['Total Sales']
    input_train, input_test, target_train, target_test = train_test_split(data_input, data_target, test_size = 0.2)
    st.write('Training dataset')
    st.write(input_train)
    st.write('Testing Dataset')
    st.write(input_test)
    model = st.selectbox('Select Model',('SVR', 'KNN','Decision Tree'))
    selectModel(model)

    st.title('User Input')

    def user_report1():
        Year = st.slider('Year',2017,2019,2017)
        Month = st.slider('Month',1,12,1)
        
        user_report1 = {
            'Year': Year,
            'Month': Month
         }

        report_data1 = pd.DataFrame(user_report1, index=[0])
        return report_data1

    user_data = user_report1()

    regressionLinear = SVR(kernel = 'linear')
    regressionLinear.fit(input_train,target_train)

    user_result = regressionLinear.predict(user_data)
    st.subheader('Your monthly sales: RM')
    output = user_result

    st.write(output)

#AINN
elif dataset_name == 'Yearly Sales Prediction':
    st.title('Yearly Sales Prediction')
    st.write('The dataset is about sales bla bla')
    st.write('')
    data = pd.read_csv("salesdata.csv")
    total = data['QUANTITYORDERED'] * data['PRICEEACH']
    data['TOTAL'] = total
    input_data = data.drop(columns = ['ORDERNUMBER','ORDERLINENUMBER','STATUS', 'QTR_ID', 'MONTH_ID', 'MSRP', 'PRODUCTLINE', 'PRODUCTCODE', 'SALES','QUANTITYORDERED','PRICEEACH'])
    
    target_data = data['SALES']
    input_train, input_test, target_train, target_test = train_test_split(input_data, target_data, test_size = 0.2)
    st.write('Training dataset')
    st.write(input_train)
    st.write('Testing Dataset')
    st.write(input_test)
    model = st.selectbox('Select Model',('SVR', 'KNN','Decision Tree'))
    selectModel(model)

    st.title('User Input')

    def user_report():
       
        YEAR_ID = st.slider('Year',2003,2005,2003)
        TOTAL = st.slider('Total Price (RM)',1,5000,1)
        user_report = {
            'YEAR_ID': YEAR_ID,
            'TOTAL': TOTAL

         }

        report_data = pd.DataFrame(user_report, index=[0])
        return report_data

    user_data = user_report()

    regressionLinear = SVR(kernel = 'linear')
    regressionLinear.fit(input_train,target_train)

    user_result = regressionLinear.predict(user_data)
    st.subheader('Yearly sales prediction: RM')
    output = user_result

    st.write(output)



    
#ZETTYYYY
elif dataset_name == 'Sales':
    st.title('Sales')
    st.write('The dataset is about sales bla bla')
    st.write('')
    data = pd.read_csv("salesdataset.csv")
    data_input = data.drop(columns = ['Store','Date','Weekly_Sales']) 
    data_target = data['Weekly_Sales']
    input_train, input_test, target_train, target_test = train_test_split(data_input, data_target, test_size = 0.2)
    st.write('Training dataset')
    st.write(input_train)
    st.write('Testing Dataset')
    st.write(input_test)
    model = st.selectbox('Select Model',('SVR', 'KNN','Decision Tree'))
    selectModel(model)

    st.title('User Input')

    def user_report():
        dept = st.slider('Dept',1,8,1)
        IsHoliday = st.checkbox('Holiday Season')
        if IsHoliday:
            IsHoliday == 'true'
        else:
            IsHoliday == 'false'

        user_report = {
            'dept': dept,
            'IsHoliday': IsHoliday
         }

        report_data = pd.DataFrame(user_report, index=[0])
        return report_data

    user_data = user_report()

    regressionLinear = SVR(kernel = 'linear')
    regressionLinear.fit(input_train,target_train)

    user_result = regressionLinear.predict(user_data)
    st.subheader('Your weekly sales: RM')
    output = user_result

    st.write(output)

else: 
    st.write('tak wujud')

    

    





