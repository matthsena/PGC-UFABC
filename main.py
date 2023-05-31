import requests
from crawler import get_top_100_articles, get_all_img, download_img

top_100_articles = get_top_100_articles()

for article in top_100_articles:
    imgs = get_all_img(article['href'])

    for img in imgs:
        download_img(img, f"{article['ranking']}-{article['text']}")