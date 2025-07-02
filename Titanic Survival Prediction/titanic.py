import streamlit as st
import pickle
import pandas as pd

# Load the trained model
with open('titanic_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Streamlit app
st.title("Titanic Survival Prediction")

# Input fields
Pclass = st.selectbox("Passenger Class (Pclass)", [1, 2, 3])
Age = st.number_input("Age", min_value=0, max_value=100, value=25)
SibSp = st.number_input("Number of Siblings/Spouses Aboard (SibSp)", min_value=0, value=0)
Parch = st.number_input("Number of Parents/Children Aboard (Parch)", min_value=0, value=0)

# Prediction
if st.button("Predict"):
    # Prepare input data
    input_data = pd.DataFrame([[Pclass, Age, SibSp, Parch]], columns=['Pclass', 'Age', 'SibSp', 'Parch'])
    prediction = model.predict(input_data)[0]
    
    # Display result
    if prediction == 1:
        st.success("The passenger is likely to survive.")
    else:
        st.error("The passenger is not likely to survive.")
