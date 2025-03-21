from extract import extract_article_content
from predict import predict_news
import streamlit as st

st.title("📰 Fake News Detection Using ML")
st.write("Enter a news article URL to predict if it's real or fake.")

url = st.text_input("Enter News Article URL:")

if st.button("Predict"):
    if url:
        content = extract_article_content(url)
        if content:
            result = predict_news(content)
            st.success(result["final"])  # Display final verdict
            st.info(result["naive_bayes"])  # Show Naive Bayes prediction
            st.info(result["cnn"])  # Show CNN prediction
        else:
            st.error("Failed to fetch article content. Please check the URL.")
    else:
        st.warning("Please enter a valid URL.")
