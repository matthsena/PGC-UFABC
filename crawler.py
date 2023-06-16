import threading
import os
import requests
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
import time

def setup_driver():
    options = webdriver.ChromeOptions()
    options.add_argument('--headless')
    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)

    return driver



def get_all_img(url):
    driver = setup_driver()
    driver.get(url)

    img_tags = driver.find_elements(By.TAG_NAME, 'img')
    img_srcs = [img.get_attribute('src') for img in img_tags]
    
    driver.quit()

    return img_srcs

def download_img(url, path):
    try:
        img_path = f'img/{path}'

        if not os.path.exists(img_path):
            os.makedirs(img_path)

        filename = url.split('/')[-1]
        
        # filter to prevent downloading svg
        if filename.split('.')[-1] == 'svg':
            return
        
        response = requests.get(url)

        with open(f'{img_path}/{url}', 'wb') as f:
            f.write(response.content)

        print(f'Successfully downloaded {filename} to img folder')
    except OSError as oserr:
        print(f'ERROR WITH {url}')

def download_images(urls, img_path):
    threads = []
    for url in urls:
        t = threading.Thread(target=download_img, args=(url, img_path))
        threads.append(t)
        t.start()

    for t in threads:
        t.join()

def get_top_100_articles():
    url = 'https://pageviews.wmcloud.org/topviews/?project=en.wikipedia.org&platform=all-access&date=2022&excludes='
    
    driver = setup_driver()
    driver.get(url)
    time.sleep(30)

    rows = driver.find_elements(By.CSS_SELECTOR, 
                                'table.output-table >'
                                'tbody.topview-entries >'
                                'tr.topview-entry')

    data = []

    for row in rows:
        try:
            ranking = row.find_element(By.CSS_SELECTOR, 
                                    'td.topview-entry--rank-wrapper >'
                                    'span.topview-entry--rank').text
            link = row.find_element(By.CSS_SELECTOR, 
                                    'td.topview-entry--label-wrapper >'
                                    'div.topview-entry--label >'
                                    'a')

            text = link.text
            href = link.get_attribute('href')

            row_data = {
                'ranking': ranking,
                'text': text,
                'href': href
            }

            data.append(row_data)
        except:
            print('Link not found')

    driver.quit()
    
    return data[:5]