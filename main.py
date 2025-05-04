import streamlit as st
import time
import firebase_admin
from firebase_admin import auth, credentials, firestore
import json
from extract import extract_article_content
from news_predict import predict_news
from visualize import animate_bar_chart, animate_pie_chart
from knowmore import show_information

# Load Firebase credentials
cred = credentials.Certificate("firebase_config.json")

# Initialize Firebase
if not firebase_admin._apps:
    cred = credentials.Certificate(firebaseConfig)
    firebase_admin.initialize_app(cred)

# Firestore client
db = firestore.client()

# Streamlit UI Config
st.set_page_config(page_title="Fake News Detection", layout="centered")
st.title("📰 Fake News Detection Using ML")

# Hide default Streamlit UI
st.markdown("""
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    </style>
""", unsafe_allow_html=True)

# Custom typewriter style
st.markdown("""
    <style>
        .typewriter-text {
            font-family: 'Helvetica', monospace;
            font-size: 1.2em;
            color: white;
            padding: 5px 0;
        }
    </style>
""", unsafe_allow_html=True)

# Session state initialization
for key in ["logged_in", "user", "username", "prediction_results", "has_results", "show_info"]:
    if key not in st.session_state:
        st.session_state[key] = False if key in ["logged_in", "has_results", "show_info"] else ""

# Typewriter animation function
def typewriter(text, delay=0.03):
    container = st.empty()
    current = ""
    for char in text:
        current += char
        container.markdown(f"<p class='typewriter-text'>{current}</p>", unsafe_allow_html=True)
        time.sleep(delay)

# Auth: Login
def login():
    st.subheader("Login")
    email = st.text_input("Email")
    password = st.text_input("Password", type="password")
    if st.button("Login"):
        try:
            user = auth.get_user_by_email(email)
            user_id = user.uid
            user_doc = db.collection("users").document(user_id).get()
            if user_doc.exists:
                username = user_doc.to_dict().get("username", "")
                if username:
                    st.session_state.logged_in = True
                    st.session_state.user = user
                    st.session_state.username = username
                    st.success(f"Welcome, {username}!")
                    st.rerun()
                else:
                    st.error("Username not found in Firestore.")
            else:
                st.error("User not found in Firestore.")
        except Exception as e:
            st.error(f"Login failed: {e}")

# Auth: Signup
def is_valid_email(email):
    return "@" in email and email.endswith(".com")

def signup():
    st.subheader("Sign Up")
    username = st.text_input("Username")
    email = st.text_input("Email")
    password = st.text_input("Password", type="password")
    if st.button("Sign Up"):
        if not is_valid_email(email):
            st.error("Invalid email! Use '@' and end with '.com'")
            return
        try:
            user = auth.create_user(email=email, password=password)
            user_id = user.uid
            db.collection("users").document(user_id).set({"username": username, "email": email})
            st.success("Account created! Please log in.")
        except Exception as e:
            st.error(f"Signup failed: {e}")

# Main App Logic
if not st.session_state.logged_in:
    option = st.selectbox("Select an option", ["Login", "Sign Up"])
    login() if option == "Login" else signup()
else:
    if st.session_state.username:
        st.subheader(f"Welcome, {st.session_state.username}!")
    url = st.text_input("Enter News Article URL:")

    if st.button("Analyze News") and url:
        content = extract_article_content(url)
        if content and "error" not in content:
            result = predict_news(content["text"])
            st.session_state.prediction_results = result
            st.session_state.has_results = True

            st.subheader("🔍 Prediction Results")
            typewriter(f"According to Naive Bayes, this news is {result.get('naive_bayes', 'N/A')}% real.")
            time.sleep(0.4)
            typewriter(f"According to SVM, this news is {result.get('svm', 'N/A')}% real.")
            time.sleep(0.4)
            typewriter(f"According to CNN, this news is {result.get('cnn', 'N/A')}% real.")
            time.sleep(0.4)
            typewriter(f"Final Verdict: This news is {result.get('final', 'N/A')}% real.")
            time.sleep(0.4)
        else:
            st.error("Failed to fetch article content. Please check the URL.")

    # Show visualizations if results exist
    if st.session_state.has_results and st.session_state.prediction_results:
        st.subheader("📊 Visualization")
        chart_type = st.radio(
            "Choose the type of chart to visualize predictions:",
            ["Bar Chart", "Pie Chart"]
        )
        if chart_type == "Bar Chart":
            animate_bar_chart(st.session_state.prediction_results)
        elif chart_type == "Pie Chart":
            animate_pie_chart(st.session_state.prediction_results)

        st.info("🟩 If the 'Final' bar is green, the news is most likely real. "
                "🟥 If red, it is most likely fake.")

    # Toggle info panel
    if st.button("Know How This Works?"):
        st.session_state.show_info = not st.session_state.show_info
        st.rerun()

    if st.session_state.show_info:
        show_information()

    if st.button("Logout"):
        for key in st.session_state.keys():
            st.session_state[key] = False if isinstance(st.session_state[key], bool) else ""
        st.rerun()
