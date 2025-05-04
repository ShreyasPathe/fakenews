import streamlit as st
from PIL import Image

def show_information():
    """
    Display comprehensive information about the fake news detection system,
    including how it works, usage instructions, algorithm descriptions, and technical details.
    """

    st.markdown("""
        <style>
            .section-title {
                font-size: 28px;
                margin-top: 2rem;
                margin-bottom: 1rem;
                font-weight: bold;
            }
            .sub-section {
                margin-left: 1rem;
                margin-bottom: 2rem;
                line-height: 1.6;
            }
            .tip-box, .warning-box {
                background-color: #1f2a3a;
                color: #f0f0f0;
                padding: 1rem;
                border-radius: 10px;
                margin-bottom: 2rem;
            }
            .tip-box {
                border-left: 5px solid #3498db;
            }
            .warning-box {
                border-left: 5px solid #e74c3c;
            }
        </style>
    """, unsafe_allow_html=True)

    # 🔍 How It Works
    st.markdown("<div class='section-title'>🔍 How It Works</div>", unsafe_allow_html=True)
    st.markdown("""
        This fake news detection system uses a combination of machine learning models to analyze the content of news articles and classify them as either **Real** or **Fake**:
        
        - 🧠 The system is trained on thousands of labeled news samples.
        - 📊 It extracts important textual features from the article and passes them to multiple classifiers.
        - 🧮 The final result is based on combining predictions from different models to ensure robustness.
    """)

    # 🤖 Algorithms Used
    st.markdown("<div class='section-title'>🤖 Algorithms Used</div>", unsafe_allow_html=True)
    st.markdown("""
        - **Naive Bayes:** A probabilistic model based on Bayes' theorem. It assumes word independence and works well for text classification due to its simplicity and speed.
        - **Support Vector Machine (SVM):** A powerful linear classifier that finds the best boundary (hyperplane) to separate real and fake news by maximizing margin.
        - **Convolutional Neural Network (CNN):** Usually used for images, but here it captures spatial patterns in text embeddings to improve classification accuracy.
    """)

    # 🛠️ How to Use
    st.markdown("<div class='section-title'>🛠️ How to Use This Website</div>", unsafe_allow_html=True)
    st.markdown("""
        1. Paste a news article **URL** into the input field.
        2. Click **Analyze** to let the models process and evaluate the news.
        3. Visualize results using **Visualize** button.
    """)

    # 📈 Confusion Matrix
    st.markdown("<div class='section-title'>📈 Confusion Matrix</div>", unsafe_allow_html=True)
    image = Image.open(r"E:\code\project\fakenews\confusion-matrix.png")
    st.image(image, caption="Confusion Matrix of the System")

    st.markdown("""
        - **True Positives (Fake-Fake):** 4417 - Fake news correctly identified.
        - **True Negatives (Real-Real):** 3957 - Real news correctly identified.
        - **False Positives (Real-Fake):** 313 - Real news misclassified as fake.
        - **False Negatives (Fake-Real):** 293 - Fake news misclassified as real.

        This matrix shows that the model performs well, with minimal misclassifications and an
        **Accuracy** of **93.67%**
    """)

    # ⚠️ Limitations
    st.markdown("<div class='section-title'>⚠️ Limitations To Be Aware Of</div>", unsafe_allow_html=True)
    st.markdown("""
        <div class='warning-box'>
            🔸 Works best with formal news articles.<br>
            🔸 May misclassify satire, parody, or opinion pieces.<br>
            🔸 Doesn't verify individual facts, only detects patterns.<br>
            🔸 Accuracy may drop for breaking or emerging news topics.
        </div>
    """, unsafe_allow_html=True)

    # 💡 Tips for Spotting Fake News
    st.markdown("<div class='section-title'>💡 Tips for Spotting Fake News</div>", unsafe_allow_html=True)
    st.markdown("""
        <div class='tip-box'>
            ✅ Check the source's credibility.<br>
            ✅ Watch out for emotional or exaggerated headlines.<br>
            ✅ Confirm with other reputable news outlets.<br>
            ✅ Look for good grammar and consistent formatting.<br>
            ✅ Verify the publication date of the story.<br>
            ✅ Investigate author profiles when possible.
        </div>
    """, unsafe_allow_html=True)
