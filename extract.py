from newspaper import Article

def extract_article_content(url):
    """Extracts title and text from a given news article URL."""
    try:
        article = Article(url)
        article.download()
        article.parse()
        return article.title + " " + article.text
    except Exception as e:
        return None  # If extraction fails
