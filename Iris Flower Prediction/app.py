import pickle
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import streamlit as st

# Function to load the model
def load_model():
    with open("iris_model.pkl", "rb") as f:
        model = pickle.load(f)
    return model

# Load the trained model
model = load_model()

# Streamlit interface
st.title("Iris Flower Species Prediction")

# Input features for prediction
sepal_length = st.number_input("Sepal Length (cm)", min_value=0.0, max_value=10.0, step=0.1)
sepal_width = st.number_input("Sepal Width (cm)", min_value=0.0, max_value=10.0, step=0.1)
petal_length = st.number_input("Petal Length (cm)", min_value=0.0, max_value=10.0, step=0.1)
petal_width = st.number_input("Petal Width (cm)", min_value=0.0, max_value=10.0, step=0.1)

# When the user clicks the button, make a prediction
if st.button("Predict"):
    # Prepare the input as a numpy array for the model
    input_features = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    
    # Perform prediction
    prediction = model.predict(input_features)
    
    # Map numerical predictions to species names
    species = {0: 'Setosa', 1: 'Versicolor', 2: 'Virginica'}
    
    st.write(f"The predicted species is: {species[prediction[0]]}")
