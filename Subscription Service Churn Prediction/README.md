# Predicting Customer Churn for a Subscription Service

## Project Description
This project focuses on building a machine learning model to predict customer churn for a subscription-based service. Customer churn occurs when customers stop using a service or cancel their subscription. The goal of this project is to identify customers who are likely to churn, which enables businesses to implement targeted retention strategies and reduce revenue loss. Retaining existing customers is often more cost-effective than acquiring new ones, making churn prediction a critical tool for improving customer loyalty.

By analyzing customer data such as subscription plans, usage patterns, and demographic information, this model predicts which customers are at risk of leaving the service.

## Project Workflow
1. **Data Collection**: 
   The dataset contains customer information including subscription plans, demographics, and service usage history.
   
2. **Data Preprocessing**:
   - Categorical columns like subscription plans (e.g., "Premium", "Basic") were one-hot encoded.
   - Numerical features were scaled using StandardScaler to normalize the input data.
   
3. **Feature Selection**: 
   Important features were selected based on feature importance from the machine learning model to reduce the dimensionality and simplify input for prediction.

4. **Model Building**:
   A machine learning model (e.g., Logistic Regression, Random Forest) was trained on historical customer data to predict whether a customer is likely to churn.
   
5. **Model Evaluation**:
   The model was evaluated using metrics such as accuracy, precision, recall, and F1-score. The goal was to maximize the model’s predictive power while avoiding overfitting.

## Features Used in the Model
- **Subscription Plan**: Type of subscription the customer is using (e.g., "Premium", "Basic").
- **Usage Frequency**: How often the customer is using the service.
- **Customer Demographics**: Age, region, and other demographic information.
- **Service Duration**: How long the customer has been subscribed.

## How to Use This Project

### Prerequisites:
1. **Install dependencies**:  
   Install all required packages from the `requirements.txt` file:
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the model**:  
   The app can be deployed using Streamlit. Run the following command in your terminal:
   ```bash
   streamlit run app.py
   ```

3. **Predict Customer Churn**:  
   Input customer features into the app to predict whether they will churn or not.

## File Structure
```
Machine-Learning-Projects/
├── Subscription Service Churn Prediction/
│   ├── app.py                 # Streamlit app for prediction
│   ├── churn_model.pkl         # Trained machine learning model
│   ├── scaler.pkl              # Scaler for preprocessing
│   ├── Model.pdf               # Model output and visualizations
│   ├── requirements.txt        # Required dependencies
```

## Technologies Used
- **Python**: For data analysis and model building.
- **Pandas**: For data manipulation.
- **Scikit-learn**: For machine learning model building.
- **Joblib**: For saving and loading the model.
- **Streamlit**: For building the app interface for predictions.

![Customer Curn Prediction](https://github.com/ritika-chaudhary-21/Machine-Learning-Projects/blob/main/Subscription%20Service%20Churn%20Prediction/Model_page-0001.jpg)
![Customer Curn Prediction](https://github.com/ritika-chaudhary-21/Machine-Learning-Projects/blob/main/Subscription%20Service%20Churn%20Prediction/Model_page-0002.jpg)
