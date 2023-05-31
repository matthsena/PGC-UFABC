import requests
from crawler import get_top_100_articles

top_100_articles = get_top_100_articles()

for article in top_100_articles:
    print(article['href'])