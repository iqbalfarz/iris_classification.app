import numpy as np # for numerical operation like linear Algebra
import pandas as pd # for data manipulation
import streamlit as st # to develop web app using pure python
import joblib as jlb # to load pickle(saved) variable/model
from PIL import Image, ImageEnhance # for Image Processing

st.title("Welcome to the Iris Flower Classification") # set the title of Project
st.image('iris_length.jpg',width=606) # set the image


st.sidebar.header('User Input Parameters')


sl = st.sidebar.slider('Sepal Length(cm)',0.0,8.0,value=2.84) # this is a widget(slider) for sepal length
sw = st.sidebar.slider('Sepal Width(cm)',0.0,4.5,value=1.34) # slider for sepal width
pl = st.sidebar.slider('Petal Length(cm)', 0.0,8.0,value=3.45) # slider for petal length
pw = st.sidebar.slider('Petal Width(cm)', 0.0,3.0,value=1.70) # slider for petal width

# user-input dataframe

user_in_df = pd.DataFrame({'Sepal Length':sl,
                           'Sepal Width':sw,
                           'Petal Length':pl,
                           'Petal Width':pw},index=[0])

st.write('<hr>',unsafe_allow_html=True)
st.text('User selected parameters...')
st.write(user_in_df)
# change all the given parameters into numpy array and reshape to one item(row).
query = np.array([sl,sw,pl,pw]).reshape(1,-1)


@st.cache
def load_model():
    model=jlb.load('model.pkl')
    return model

# loading the already saved model(refer: goto model.py for further detail)
model = load_model()

prediction = model.predict(query) # find prediction of query point

st.write('<hr>',unsafe_allow_html=True)
st.write('## Output of selected parameters...') # write something on the user console

@st.cache # caching the images
def load_images(): 
    virginica = Image.open('virginica.jpg').resize((200,200))
    versicolor = Image.open('versicolor.jpg').resize((200,200))
    setosa = Image.open('setosa.jpg').resize((200,200))
    return virginica, versicolor, setosa

virginica, versicolor,setosa = load_images() # loading the images

vir_i = 0.5 # virginica intensity
ver_i = 0.5 # versicolor intensity
set_i = 0.5 # setosa intensity

if prediction=='virginica':
    vir_i = 1.1
elif prediction=='versicolor':
    ver_i = 1.1
elif prediction=='setosa':
    set_i = 1.1
    
    
enhancer_vir = ImageEnhance.Brightness(virginica)
enhancer_ver = ImageEnhance.Brightness(versicolor)
enhancer_set = ImageEnhance.Brightness(setosa)

ed_vir = enhancer_vir.enhance(vir_i)
ed_ver = enhancer_ver.enhance(ver_i)
ed_set = enhancer_set.enhance(set_i)

st.image([ed_vir,ed_ver,ed_set],caption=['virginica','versicolor','setosa'],width=200)
    
    
@st.cache # csv data is cached
def load_data():
    data=pd.read_csv('iris.csv')
    return data


st.write('<hr>',unsafe_allow_html=True)
# to see the raw data I used checkbox to be checked...
if st.checkbox('Want to see the raw data...check the box'):
    data = load_data()
    st.dataframe(data.style.highlight_max(axis=0))

    
    
