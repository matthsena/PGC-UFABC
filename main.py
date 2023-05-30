import requests

# URL of the Wikipedia API
url = 'https://pt.wikipedia.org/w/api.php'

# Parameters for the API request
params = {
    'action': 'query',
    'prop': 'extracts',
    'titles': 'The_Boys_(TV_series)',
    'format': 'json',
    'explaintext': True,
    'exsectionformat': 'wiki',
    'exsentences': 5
}

# Send the API request
response = requests.get(url, params=params)

# Get the page content from the API response
page = next(iter(response.json()['query'].values()))

# Print the page content
print(response.json)