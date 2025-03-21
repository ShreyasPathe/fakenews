import re

def preprocess_text(text):
    """Cleans text by removing special characters and converting to lowercase."""
    text = re.sub(r'[^a-zA-Z0-9]', ' ', text)  # Remove special characters
    text = text.lower()  # Convert to lowercase
    return text
