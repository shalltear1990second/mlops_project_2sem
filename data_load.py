import requests
from urllib.parse import urlencode

DATA_RAW_SOURCE_PATH = \
    "./data_raw_source/car.csv"

base_url = 'https://cloud-api.yandex.net/v1/disk/public/resources/download?'
public_key = 'https://disk.yandex.ru/d/buYprzTIVN8vbA'

final_url = base_url + urlencode(dict(public_key=public_key))
response = requests.get(final_url)
download_url = response.json()['href']

download_response = requests.get(download_url)
with open(DATA_RAW_SOURCE_PATH, 'wb') as f:
    f.write(download_response.content)

print("data_load - успешно выполнен")
