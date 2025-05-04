from newspaper import Article

def extract_article_content(url):
    """Extracts title and text from a given news article URL."""
    try:
        article = Article(url)
        article.download()
        article.parse()
        
        if not article.text.strip():  # Check if content is empty
            return {"error": "Failed to extract article content."}
        
        return {"title": article.title, "text": article.text}
    
    except Exception as e:
        return {"error": f"Extraction failed: {str(e)}"}  # Return error message
