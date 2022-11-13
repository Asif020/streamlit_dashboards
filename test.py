import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# app ke heading
st.write(""""
# Explore different ML models and datasets
Daikhty hn kon se best hn  may say?
""")

# data set name ak box may daal  k sidebar pay laga do 
dataset_name = st.sidebar.selectbox(
    'select Dataset',
    ('Iris', 'Breast Cnacer', 'Wine')
)

# or isi k nichay Classifier k nam ak dabay may dal do
classifier_name = st.sidebar.selectbox(
    'Select classifier',
    ('KNN', 'SVM', 'RandomForest')
)

# ab hum nay ak function define kena hai dataset ko load krny k laiy

def get_dataset(dataset_name):
    data = None
    if dataset_name =="Iris":
        data = datasets.load_iris()
    elif dataset_name == "Wine":
        data = datasets.load_wine()
    else:
        data = datasets.load_breast_cancer()
    X = data.data
    y = data.target
    return X, y

    # ab is function ko bula lay gayn or X, y variable k equal rakh layn gay
    X,y = get_function(dataset_name)

# ab hum apnay data set ki shape ko ap pay print kr dayn gay
st.write('Shape of dataset:') #X.shape)
st.write('number of classes:', len(np.unique))