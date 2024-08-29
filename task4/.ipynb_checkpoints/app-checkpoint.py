import streamlit as st
import joblib
import nltk
from nltk.tokenize import word_tokenize

# Load models and vectorizer
nb_model = joblib.load('naive_bayes_model.pkl')
lr_model = joblib.load('logistic_regression_model.pkl')
svm_model = joblib.load('svm_model.pkl')
tfidf_vectorizer = joblib.load('tfidf_vectorizer.pkl')

# Preprocessing function
nltk.download('punkt')
stop_words = set(nltk.corpus.stopwords.words('english'))

def preprocess_text(text):
    words = word_tokenize(text.lower())
    words = [word for word in words if word.isalnum() and word not in stop_words]
    return ' '.join(words)

# Function to predict genre
def predict_genre(model, plot_summary):
    plot_summary_preprocessed = preprocess_text(plot_summary)
    plot_summary_tfidf = tfidf_vectorizer.transform([plot_summary_preprocessed])
    predicted_genre = model.predict(plot_summary_tfidf)
    return predicted_genre[0]

# Streamlit UI
st.title('Movie Genre Prediction')

plot_summary = st.text_area("Enter the plot summary:")

if st.button("Predict Genre"):
    predicted_genre_nb = predict_genre(nb_model, plot_summary)
    predicted_genre_lr = predict_genre(lr_model, plot_summary)
    predicted_genre_svm = predict_genre(svm_model, plot_summary)
    
    st.write(f"Predicted genre (Naive Bayes): {predicted_genre_nb}")
    st.write(f"Predicted genre (Logistic Regression): {predicted_genre_lr}")
    st.write(f"Predicted genre (SVM): {predicted_genre_svm}")
