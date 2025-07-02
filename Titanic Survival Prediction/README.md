# 🚢 Titanic Survival Prediction

This project uses machine learning to predict passenger survival on the Titanic using the famous [Titanic dataset](https://www.kaggle.com/c/titanic). It involves data preprocessing, feature selection, and classification using models such as Random Forest.

## 📌 Project Overview

The goal is to build a predictive model that answers the question:  
**“What sorts of people were more likely to survive?”**

### ✅ Key Steps:
- Data Cleaning & Imputation
- Feature Engineering (e.g., Family Size, Title)
- Feature Selection with \`SelectKBest\`
- Model Training with \`RandomForestClassifier\`
- Evaluation with accuracy metric

---

## 🧪 Dependencies

Make sure you have the following Python packages installed:

<pre>bash
pip install pandas numpy scikit-learn matplotlib seaborn
</pre>

---

## 🧰 File Structure

<pre>
├── titanic.ipynb          # Main Jupyter notebook with all code
├── Titanic-Dataset.csv
├── titanic.py
├── titanic_model.pkl
├── README.md              # Project overview and instructions
</pre>

---

## 🚀 How to Run

1. Clone the repo:
<pre>bash
git clone http://github.com/ritika-chaudhary-21/Machine-Learning-Projects/tree/main/Titanic%20Survival%20Prediction
cd titanic-survival-prediction
</pre>

3. Run the notebook:
Open \`titanic.ipynb\` in Jupyter Lab/Notebook or VS Code.

---

## 🔍 Model Pipeline

We use a Scikit-learn \`Pipeline\` to streamline preprocessing and feature selection:

<pre> python
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.ensemble import RandomForestClassifier

pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('feature_selection', SelectKBest(score_func=f_classif, k=10)),
    ('classifier', RandomForestClassifier())
]) </pre>

---

## 📊 Results

- **Model Accuracy:** ~68% (baseline)
- Improved by:
  - Feature engineering
  - Hyperparameter tuning
  - Proper data imputation and selection
![image](https://github.com/user-attachments/assets/be472f09-9799-4c11-acef-aac7251f262e)

---

## 📈 Future Work

- Use ensemble models (XGBoost, LightGBM)
- Deploy model with Streamlit
- Perform cross-validation and ROC analysis
- Tune hyperparameters using \`GridSearchCV\`

---
