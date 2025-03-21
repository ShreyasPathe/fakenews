import numpy as np
import joblib
from tensorflow.keras.preprocessing.sequence import pad_sequences
from preprocess import preprocess_text
from tensorflow.keras.models import load_model

# Load trained models
naive_bayes_model = joblib.load('naive_bayes_model.pkl')  # Naive Bayes
cnn_model = load_model('cnn_model.h5')  # CNN
tokenizer = joblib.load('tokenizer.pkl')  # Tokenizer for CNN
vectorizer = joblib.load('tfidf_vectorizer.pkl')  # TF-IDF Vectorizer

def predict_news(text):
    """Predicts if news is Fake or Real using Naive Bayes and CNN models with probability scores."""
    
    # Step 1: Preprocess the input text
    processed_text = preprocess_text(text)
    
    # Step 2: Convert text to TF-IDF vector (required for Naive Bayes)
    text_tfidf = vectorizer.transform([processed_text])  # Transform input text into TF-IDF vector

    # Step 3: Naive Bayes Probability Prediction
    nb_probs = naive_bayes_model.predict_proba(text_tfidf)[0]  # Probabilities [Real, Fake]
    nb_real_prob = nb_probs[0] * 100  # Probability of being Real (in percentage)

    # Step 4: CNN Probability Prediction
    seq = tokenizer.texts_to_sequences([processed_text])
    padded_seq = pad_sequences(seq, maxlen=300)  # Match CNN input size
    cnn_real_prob = (1 - cnn_model.predict(padded_seq)[0][0]) * 100  # Convert Fake probability to Real percentage

    # Step 5: **Final Weighted Average Prediction**
    final_real_prob = (nb_real_prob + cnn_real_prob) / 2  # Average probability of being Real

    # Round percentages
    nb_real_prob = round(nb_real_prob, 2)
    cnn_real_prob = round(cnn_real_prob, 2)
    final_real_prob = round(final_real_prob, 2)

    # Step 6: Display Results
    return {
        "naive_bayes": f"1️⃣ According to Naive Bayes, this news is **{nb_real_prob}% real**.",
        "cnn": f"2️⃣ According to CNN, this news is **{cnn_real_prob}% real**.",
        "final": f"3️⃣ **Final Verdict: This news is {final_real_prob}% real.**"
    }
