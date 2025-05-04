import numpy as np
import joblib
from tensorflow.keras.preprocessing.sequence import pad_sequences
from preprocess import preprocess_text
from tensorflow.keras.models import load_model

# Load trained models
naive_bayes_model = joblib.load("naive_bayes_model.pkl")
svm_model = joblib.load("svm_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")
cnn_model = load_model("cnn_model.h5")  
tokenizer = joblib.load("tokenizer.pkl")

def predict_news(text):
    processed_text = preprocess_text(text)
    text_tfidf = vectorizer.transform([processed_text])
    
    nb_probs = naive_bayes_model.predict_proba(text_tfidf)[0]
    nb_real_prob = nb_probs[1] * 100
    
    svm_probs = svm_model.predict_proba(text_tfidf)[0]
    svm_real_prob = 100 - svm_probs[1] * 100
    
    seq = tokenizer.texts_to_sequences([processed_text])
    padded_seq = pad_sequences(seq, maxlen=300)
    cnn_raw = cnn_model.predict(padded_seq)[0][0]
    cnn_real_prob = float((1 - cnn_raw) * 100)
    
    final_real_prob = ((0.93 * nb_real_prob) + (0.998 * cnn_real_prob) + (0.989 * svm_real_prob)) / (0.93 + 0.998 + 0.989)
    
    # Round probabilities
    nb_real_prob = round(nb_real_prob, 2)
    svm_real_prob = round(svm_real_prob, 2)
    cnn_real_prob = round(cnn_real_prob, 2)
    final_real_prob = round(final_real_prob, 2)
    
    # Make prediction labels
    nb_label = "REAL" if nb_real_prob >= 50 else "FAKE"
    svm_label = "REAL" if svm_real_prob >= 50 else "FAKE"
    cnn_label = "REAL" if cnn_real_prob >= 50 else "FAKE"
    final_label = "REAL" if final_real_prob >= 50 else "FAKE"
    
    # Return the results with both numeric values and formatted strings
    return {
        "naive_bayes": nb_real_prob,
        "svm": svm_real_prob,
        "cnn": cnn_real_prob,
        "final": final_real_prob,
        "nb_label": nb_label,
        "svm_label": svm_label,
        "cnn_label": cnn_label,
        "final_label": final_label,
        "naive_bayes_text": f"According to Naive Bayes, this news is {nb_real_prob}% real.",
        "svm_text": f"According to SVM, this news is {svm_real_prob}% real.",
        "cnn_text": f"According to CNN, this news is {cnn_real_prob}% real.",
        "final_text": f"This news is {final_real_prob}% real."
    }
