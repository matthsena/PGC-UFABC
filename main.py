import concurrent.futures
import re
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
import os
import urllib.request
import pandas as pd
from PIL import Image

def is_gif(filename):
    return filename.split('.')[-1] == 'gif'

def gif_to_png(filename):
    with Image.open(filename) as im:
    
        im.seek(im.n_frames - 1)
    
        png_filename = filename.replace('.gif', '.png')
        im.save(png_filename, 'png')
        return png_filename

def check_and_convert_image(filename):
    if is_gif(filename):
        print(f'Convertando {filename} para PNG')
        png_filename = gif_to_png(filename)

    

        return png_filename
    else:
        return filename

def replace_px_value(string):
    modified_string = re.sub(r'\b\d+px', '2240px', string)
    return modified_string

# url = 'https://en.wikipedia.org/wiki/Pel%C3%A9'
# output_directory = './en-pele'

def download_image(img_link, output_directory):
    pattern = r'/(\d+)px-'
    matches = re.findall(pattern, img_link)

    if len(matches):
        if int(matches[0]) < 100:
            print(f'imagem ignorada: {img_link}')
        else:
            url_img = replace_px_value(img_link)
            filename = url_img.split('/')[-1]

            if not filename.split('.')[-1] == 'svg':
                urllib.request.urlretrieve(url_img, os.path.join(output_directory, filename))
                print(f'IMAGEM BAIXADA: {filename}')

                check_and_convert_image(os.path.join(output_directory, filename))
    else:
        print(f'imagem ignorada: {img_link}')

def start_download(url, output_directory):
    options = Options()
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--headless")

    webdriver_service = Service(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=webdriver_service, options=options)
    driver.get(url)

    WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.TAG_NAME, "img")))

    img_elements = driver.find_elements(By.TAG_NAME, "img")
    img_links = [img.get_attribute('src') for img in img_elements]

    if not os.path.isdir(output_directory):
        os.makedirs(output_directory)

    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = []
        for img_link in img_links:
            futures.append(executor.submit(download_image, img_link, output_directory))

        for future in concurrent.futures.as_completed(futures):
            pass

    driver.quit()

df = pd.read_csv('dataset.csv')

for i, row in df.iterrows():
    img_path = f"data/{row['title']}/{row['lang']}"

    if row['url'] != '':
        start_download(row['url'], img_path)
    else:
        if not os.path.isdir(img_path):
            os.makedirs(img_path)