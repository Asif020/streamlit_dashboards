import streamlit as st
import seaborn as sns
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, r2_score

# make containers
header = st.container()
data_sets = st.container()
features = st.container()
model_training = st.container()

with header:
    st.title("Kashti ki app")
    st.text("In the projrct we will work on kashti data")

with data_sets:
        st.header("kashti doob gaye, Haww!")
        st.text("We will work with titanic dataset")
        #import data
        df = sns.load_dataset('titanic')
        df = df.dropna()
        st.write(df.head(10))
      

        st.subheader('Sambha, Ara oooh sambha, kitnay aadmi thay?')
        st.bar_chart(df['sex'].value_counts())

        # other plot
        st.subheader('class ka hisaab sa faraq')
        st.bar_chart(df['class'].value_counts())
       #barplot
        st.bar_chart(df['age'].sample(10)) #or head()

        with features:
            st.header("These are our app features")
            st.text("Awen bht sary features add kartay hyn, asaan hi hy")
            st.markdown('1.**Feature 1:** This will tell us pata nhai')
            st.markdown('2.**Feature 2:** This will tell us pata nhai')

with model_training:
    st.header("kashti walo ka kia bna?-Model training")
    st.text("In the projrct we will work on kashti data")

# making columns
input, display = st.columns(2)

# pehlay columns main ap k selections points hun
max_depth = input.slider("How many pepole do you know?", min_value=10, max_value= 100, value = 20, step=5)

# n estimators
n_estimators = input._selectbox("How many tree should be there in a RF?,", options=[50,100,300,'No limit'])

# adding list of features
input.write(df.columns)


# input features from user

input_features = input.text_input('Which feature we should use?')


# machine learning model
model = RandomForestRegressor(max_depth=max_depth, n_estimators=n_estimators)
# yahan per ham aik conditions lagaye gay
if n_estimators == 'No limit':
    model = RandomForestRegressor(max_depth=max_depth)
else:
    model = RandomForestRegressor(max_depth=max_depth, n_estimators=n_estimators)


# Define X and y
X = df[[input_features]]
y = df[['fare']]

# fit our model
model.fit(X,y)
pred = model.predict(y)


# Display metrices
display.subheader("Mean absolute error of the model is: ")
display.write(mean_absolute_error(y,pred))
display.subheader("Mean squred error of the model is: ")
display.write(mean_absolute_error(y,pred))
display.subheader("R squred score of the model is: ")
display.write(r2_score(y,pred))


        