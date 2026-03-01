import streamlit as st
import pandas as pd
import numpy as np

## Title
st.title("Laalalalalalalala")

st.write("textttttt")

# create dataframe
df = pd.DataFrame({
    'First col ': [1,2,3],
    'Second col': [10,20,30]
})
st.write(df)


#create line chart

chartData = pd.DataFrame(
    np.random.randn(20,3), columns=['a', 'b', 'c']
)

st.line_chart(chartData)

age = st.slider("Age: ",0,100,25)
st.write(f"Your age: {age}")

name = st.text_input("Enter name: ")
if name:
    st.write(f"Hello {name}")




options = ["Python", "Java", "C++", "JavaScript"]
choice = st.selectbox("Choose your favorite language:", options)
st.write(f"You selected {choice}.")

if name:
    st.write(f"Hello, {name}")


data = {
    "Name": ["John", "Jane", "Jake", "Jill"],
    "Age": [28, 24, 35, 40],
    "City": ["New York", "Los Angeles", "Chicago", "Houston"]
}

df = pd.DataFrame(data)
df.to_csv("sampledata.csv")
st.write(df)


uploaded_file=st.file_uploader("Choose a CSV file",type="csv")

if uploaded_file is not None:
    df=pd.read_csv(uploaded_file)
    st.write(df)