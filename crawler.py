from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
import time


def get_top_100_articles():
    # Setup driver
    options = webdriver.ChromeOptions()
    options.add_argument('--headless')
    options.add_argument('--disable-gpu')
    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)

    # URL to scrape
    url = 'https://pageviews.wmcloud.org/topviews/?project=en.wikipedia.org&platform=all-access&date=2022&excludes='

    driver.get(url)

    # Wait for the page to fully load
    time.sleep(10)

    # Find the table
    rows = driver.find_elements(By.CSS_SELECTOR, 
                                'table.output-table >'
                                'tbody.topview-entries >'
                                'tr.topview-entry')

    # List of articles
    data = []

    for row in rows:
        try:
            # Get page ranking
            ranking = row.find_element(By.CSS_SELECTOR, 
                                    'td.topview-entry--rank-wrapper >'
                                    'span.topview-entry--rank').text
            # Get the link in each row
            link = row.find_element(By.CSS_SELECTOR, 
                                    'td.topview-entry--label-wrapper >'
                                    'div.topview-entry--label >'
                                    'a')

            # Get the text and href attribute of the link
            text = link.text
            href = link.get_attribute('href')

            # Print the text and href
            # Create a dictionary with the data
            row_data = {
                'ranking': ranking,
                'text': text,
                'href': href
            }

            # Append the dictionary to the list
            data.append(row_data)
        except:
            # Handle case where link is not found
            print('Link not found')

    # Close the driver
    driver.quit()

    return data

print(get_top_100_articles())