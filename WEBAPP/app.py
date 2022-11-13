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
    ('KNN', 'SVM', 'Forest')
)