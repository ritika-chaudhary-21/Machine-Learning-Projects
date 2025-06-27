# 🎬 Movie Genre Classifier

This project is a machine learning-based web app that predicts the **genre(s)** of a movie based on its **plot description**. It's built using **Scikit-learn**, **NLTK**, and deployed with **Streamlit** for an interactive user interface.

---

## 🚀 Features

- ✅ Multi-label genre classification (movies can belong to multiple genres)
- ✅ Text preprocessing with NLTK (lemmatization + stopword removal)
- ✅ Feature extraction using **TF-IDF**
- ✅ Trained and evaluated using models like:
  - Logistic Regression (OneVsRest)
  - Naive Bayes
  - Random Forest (optional)
- ✅ Custom threshold-based prediction to ensure better genre coverage
- ✅ Interactive web interface with **Streamlit**

---

## 📁 Dataset

The dataset is from Kaggle: [IMDb Movie Genre Dataset]((https://www.kaggle.com/datasets/hijest/genre-classification-dataset-imdb))

- **Train file**: Contains `ID ::: TITLE ::: GENRE ::: DESCRIPTION`
- **Test file**: Contains `ID ::: TITLE ::: DESCRIPTION`
- Genres are multi-label and derived from movie metadata

---

## 📊 ML Pipeline

1. **Data Preprocessing**
   - Extract and clean plot descriptions
   - Convert genre strings to lists for multi-label handling

2. **Text Processing**
   - Lowercase conversion, punctuation removal
   - Stopword removal and lemmatization (via NLTK)
   - TF-IDF vectorization (`max_features=5000`)

3. **Model Training**
   - Multi-label classification using `OneVsRestClassifier`
   - Compared Logistic Regression, Naive Bayes, Random Forest

4. **Evaluation Metrics**
   - **Hamming Loss**
   - **F1 Score (macro/micro)**
   - **Classification Report (per genre)**

5. **Threshold Optimization**
   - Used `predict_proba()` with threshold tuning (e.g., `0.3`)
   - Fallback mechanism: ensure at least one genre is always predicted

---

## 🖥️ Streamlit App

### Run Locally

```bash
pip install -r requirements.txt
streamlit run app.py
