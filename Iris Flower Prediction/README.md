# Iris Flower Classification Project

### Overview

The **Iris Flower Classification** project is a simple yet powerful machine learning application that predicts the species of Iris flowers based on their features. This project uses the famous Iris dataset, which contains three species of Iris flowers: **Setosa**, **Versicolor**, and **Virginica**. The model is trained using **RandomForestClassifier**, a powerful and efficient ensemble algorithm.

The goal of this project is to classify flowers based on the length and width of their sepals and petals. The project is developed with the following technologies: **Python**, **scikit-learn**, and **Streamlit** for the web interface.

### Project Structure
````
iris-classification/
│
├── iris project.ipynb  # Jupyter Notebook for model training and analysis
├── iris_model.pkl      # Serialized trained model (pickle file)
├── app.py              # Streamlit application for model deployment
└── README.md           # Project description and usage guide
````
### Dataset
The Iris dataset contains 150 samples with the following features:
- **Sepal Length** (cm)
- **Sepal Width** (cm)
- **Petal Length** (cm)
- **Petal Width** (cm)
- **Species** (Setosa, Versicolor, or Virginica)

You can download the dataset from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/iris).

### Model
The project utilizes a **Random Forest Classifier** for the task of flower species classification. Random forests are an ensemble learning method that operates by constructing multiple decision trees and merging their results to improve accuracy and avoid overfitting.

### Features
- **User Interface**: A web-based user interface built with **Streamlit** allows users to input the flower measurements (sepal length, sepal width, petal length, petal width) and predict the species.
- **Model Persistence**: The trained model is saved as a **pickle file** (`iris_model.pkl`), which is loaded into the web application for predictions.
- **Accuracy**: The model has an accuracy of **93%** based on testing data.

### How to Use
1. **Clone the repository**:
   ```bash
   git clone <repository_url>
   cd iris-classification
   ```

2. **Install dependencies**:
   Install the required Python libraries using `pip`:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Streamlit app**:
   Use the following command to start the web application:
   ```bash
   streamlit run app.py
   ```

4. **Input flower features**:
   - Enter values for the flower's sepal length, sepal width, petal length, and petal width.
   - Click on the "Predict" button to get the predicted species.

### Dependencies

- **Python 3.x**
- **scikit-learn** (>=1.5.1)
- **numpy**
- **Streamlit**

### Model Training (Optional)

The model can be retrained using the provided Jupyter notebook (`iris project.ipynb`). Open the notebook in Jupyter and run the cells to train the model and save it as `iris_model.pkl`.

### Example

Here’s a quick example of how the application works:
1. Input flower measurements:  
   - Sepal Length: 5.1  
   - Sepal Width: 3.5  
   - Petal Length: 1.4  
   - Petal Width: 0.2  

2. Click on **Predict**.

3. The app will predict the species as **Setosa**.

![Iris Flower Prediction](https://github.com/ritika-chaudhary-21/Machine-Learning-Projects/blob/main/Iris%20Flower%20Prediction/Model.png?raw=true)
