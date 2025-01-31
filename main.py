"""
- use vgg foundational model for now
- fine tune with cross referenced eonet wildfire events and copernicus api
- general satellite imagery also available from copernicus api

- preprocess
- train
- validation
- test
- export parameters

"""

# import tensorflow as tf
# import cv2 as cv
# import os
import requests
import json
import pandas
# import onnxruntime

def retrieve():
    storage = {}
    wildfire_url = "https://eonet.gsfc.nasa.gov/api/v3/categories/wildfires"
    response = requests.get(url=wildfire_url)
    data = response.json()
    with open('categories.json', 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

if __name__ == "__main__":
    retrieve()
