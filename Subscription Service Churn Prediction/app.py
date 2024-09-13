import streamlit as st
import joblib
import numpy as np
import pandas as pd  # Import pandas

# Load the model and preprocessor
model = joblib.load('simple_churn_model.pkl')
preprocessor = joblib.load('preprocessor.pkl')  # Ensure this includes both scaling and encoding

# Streamlit app
st.title("Churn Prediction")

# Numeric input fields
account_age = st.number_input("Account Age (months)", min_value=0, max_value=100, value=12)
monthly_charges = st.number_input("Monthly Charges", min_value=0.0, value=70.50)
total_charges = st.number_input("Total Charges", min_value=0.0, value=850.25)
viewing_hours_per_week = st.number_input("Viewing Hours Per Week", min_value=0.0, value=15.0)
average_viewing_duration = st.number_input("Average Viewing Duration (minutes)", min_value=0.0, value=45.0)
content_downloads_per_month = st.number_input("Content Downloads Per Month", min_value=0, value=5)
user_rating = st.number_input("User Rating", min_value=0.0, max_value=5.0, value=4.5)
support_tickets_per_month = st.number_input("Support Tickets Per Month", min_value=0, value=1)
watchlist_size = st.number_input("Watchlist Size", min_value=0, value=10)

# Categorical input fields
subscription_type = st.selectbox("Subscription Type", ["Basic", "Standard", "Premium"])
payment_method = st.selectbox("Payment Method", ["Credit Card", "Debit Card", "PayPal", "Other"])
paperless_billing = st.selectbox("Paperless Billing", ["Yes", "No"])
content_type = st.selectbox("Content Type", ["Movies", "Series", "Both"])
multi_device_access = st.selectbox("Multi-Device Access", ["Yes", "No"])
device_registered = st.selectbox("Device Registered", ["Yes", "No"])
genre_preference = st.selectbox("Genre Preference", ["Action", "Drama", "Comedy", "Other"])
gender = st.selectbox("Gender", ["Male", "Female", "Other"])
parental_control = st.selectbox("Parental Control", ["Enabled", "Disabled"])
subtitles_enabled = st.selectbox("Subtitles Enabled", ["Yes", "No"])

if st.button("Predict"):
    try:
        # Compile all features into a dictionary
        input_features = {
            'AccountAge': account_age,
            'MonthlyCharges': monthly_charges,
            'TotalCharges': total_charges,
            'ViewingHoursPerWeek': viewing_hours_per_week,
            'AverageViewingDuration': average_viewing_duration,
            'ContentDownloadsPerMonth': content_downloads_per_month,
            'UserRating': user_rating,
            'SupportTicketsPerMonth': support_tickets_per_month,
            'WatchlistSize': watchlist_size,
            'SubscriptionType': subscription_type,
            'PaymentMethod': payment_method,
            'PaperlessBilling': paperless_billing,
            'ContentType': content_type,
            'MultiDeviceAccess': multi_device_access,
            'DeviceRegistered': device_registered,
            'GenrePreference': genre_preference,
            'Gender': gender,
            'ParentalControl': parental_control,
            'SubtitlesEnabled': subtitles_enabled
        }

        # Convert input features to a DataFrame
        input_data = pd.DataFrame([input_features])  # Single-row DataFrame

        # Use the preprocessor to transform the input correctly
        input_data_processed = preprocessor.transform(input_data)  # Preprocessor handles encoding and scaling

        # Make prediction
        prediction = model.predict(input_data_processed)
        st.write(f"Prediction: {'Churn' if prediction[0] == 1 else 'No Churn'}")
    except Exception as e:
        st.error(f"Error in input processing: {e}")
