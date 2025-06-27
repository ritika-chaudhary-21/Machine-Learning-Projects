import streamlit as st
import joblib
import nltk
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


# Load saved model components
model = joblib.load('best_model.pkl')            # Your trained classifier
vectorizer = joblib.load('tfidf_vectorizer.pkl') # Your trained TF-IDF vectorizer
mlb = joblib.load('mlb.pkl')                     # MultiLabelBinarizer

# Download resources (this part is crucial!)
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    words = text.split()
    words = [lemmatizer.lemmatize(w) for w in words if w not in stop_words]
    return ' '.join(words)


# Streamlit UI
st.title("🎬 Movie Genre Classifier")
st.write("Enter a movie plot and get the predicted genre(s)")

input_text = st.text_area("Movie Plot Description")

if st.button("Predict Genre"):
    if input_text.strip() == "":
        st.warning("Please enter a plot description.")
    else:
        cleaned_text = preprocess_text(input_text)
        vectorized_text = vectorizer.transform([cleaned_text])
        probs = model.predict_proba(vectorized_text)
        threshold = 0.3  # try 0.2–0.4 depending on results
        predicted_labels = (probs >= threshold).astype(int)
        genres = mlb.inverse_transform(predicted_labels)

        if genres[0]:
            st.success(f"🎯 Predicted Genres: {', '.join(genres[0])}")
        else:
            st.info("No genres predicted.")
