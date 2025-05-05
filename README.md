# Fake News Detection Using Machine Learning

This project leverages machine learning models to analyze and predict whether a news article is real or fake. The app allows users to log in, submit news article URLs, and receive predictions on the veracity of the news based on multiple ML models: Naive Bayes, SVM, and CNN. It also visualizes the results using interactive charts.

## Features
- **Login/Sign Up**: Users can log in or sign up using Firebase Authentication.
- **URL Submission**: Users can input a URL of a news article to analyze.
- **Prediction Results**: The app uses three machine learning models to determine the likelihood of the news being real or fake.
- **Data Visualization**: Interactive bar or pie charts to visualize the prediction results.
- **Information Panel**: A section explaining how the app works and the underlying ML models.
- **Logout**: Allows users to log out from the app.

## Installation

### Prerequisites:
1. Python 3.x
2. Firebase Project and Firebase Credentials (stored in `firebase_config.json`)

### Steps to Run:

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/fake-news-detection.git

2. Navigate into the project directory:

   ```bash
   cd fake-news-detection
   ```

3. Install required dependencies:

   ```bash
   pip install -r requirements.txt
   ```

4. Add your `firebase_config.json` file to the project directory. You can obtain this file by creating a Firebase project and generating a service account key from the Firebase console.

5. Run the Streamlit app:

   ```bash
   streamlit run app.py
   ```

6. The app should now be accessible on your browser at `http://localhost:8501`.

## Firebase Configuration

* This app requires Firebase Authentication and Firestore to store and manage user data.
* The `firebase_config.json` file contains sensitive Firebase credentials. **Ensure it is not shared publicly or used for illegal/unauthorized access**.

## Project Structure

* `app.py`: Main Streamlit app file that runs the user interface and app logic.
* `firebase_config.json`: Firebase service account credentials (should not be shared or committed to version control).
* `extract.py`: Script for extracting content from news articles using web scraping.
* `news_predict.py`: Script that houses the machine learning models to predict fake news.
* `visualize.py`: Script for generating animated charts to visualize the prediction results.
* `knowmore.py`: Provides an information panel explaining the app and the ML models used.

## For Issues

If you encounter any issues or need further assistance, feel free to reach out at:
**Email**: [patheshreyas7@gmail.com](mailto:patheshreyas7@gmail.com)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
