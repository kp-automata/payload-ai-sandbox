"""Sandbox code before actual development
- use vgg foundational model for now
- fine tune with cross referenced eonet wildfire events and copernicus api
- general satellite imagery also available from copernicus api

- preprocess
- train
- validation
- test
- export parameters

"""
import cv2
import os
import time
import requests
import json
import onnxruntime
import numpy as np
import onnx
import os
import glob
import tensorflow as tf
import argparse

from selenium import webdriver
from selenium.webdriver.common.by import By

from urllib.parse import urljoin
from bs4 import BeautifulSoup
from onnx import numpy_helper as nh

def create_image_net_model_predictions(model_type):
    for image in os.listdir("data"):
        print(image)
        loaded = tf.keras.preprocessing.image.load_img(f"data/{image}", target_size=(224, 224))
        preprocessed = np.expand_dims(tf.keras.preprocessing.image.img_to_array(loaded), axis=0)

        if model_type == "vgg":
            to_predict = tf.keras.applications.vgg16.preprocess_input(preprocessed)
            model = tf.keras.applications.vgg16.VGG16(weights='imagenet', include_top=True)

        elif model_type == "resnet":
            to_predict = tf.keras.applications.resnet50.preprocess_input(preprocessed)
            model = tf.keras.applications.resnet50.ResNet50(weights='imagenet', include_top=True)

        else:
            return

        predictions = model.predict(to_predict)

        if model_type == "vgg":
            print('Keras Predicted:', tf.keras.applications.vgg16.decode_predictions(predictions, top=3)[0])

        elif model_type == "resnet":
            print('Keras Predicted:', tf.keras.applications.resnet50.decode_predictions(predictions, top=3)[0])

        model.export(os.path.join("export/", model.name))

def onnx_vgg_min_model_test():
    test_data_dir = 'vgg16/test_data_set_0'
    # Load inputs
    inputs = []
    inputs_num = len(glob.glob(os.path.join(test_data_dir, 'input_*.pb')))
    for i in range(inputs_num):
        input_file = os.path.join(test_data_dir, 'input_{}.pb'.format(i))
        tensor = onnx.TensorProto()
        with open(input_file, 'rb') as f:
            tensor.ParseFromString(f.read())
        inputs.append(nh.to_array(tensor))

    # Load reference outputs
    ref_outputs = []
    ref_outputs_num = len(glob.glob(os.path.join(test_data_dir, 'output_*.pb')))
    for i in range(ref_outputs_num):
        output_file = os.path.join(test_data_dir, 'output_{}.pb'.format(i))
        tensor = onnx.TensorProto()
        with open(output_file, 'rb') as f:
            tensor.ParseFromString(f.read())
        ref_outputs.append(nh.to_array(tensor))

    session = onnxruntime.InferenceSession('vgg16/vgg16.onnx')
    outputs = session.run(None, {'data': inputs[0]})
    print(outputs)

    # # Compare the results with reference outputs.
    for ref_o, o in zip(ref_outputs, outputs):
        np.testing.assert_almost_equal(ref_o, o)

def test_basic_image():
    # TODO: implement for custom ops
    # Load image
    image = preprocess_image('data/your_image.jpg')
    # Run inference
    session = onnxruntime.InferenceSession('vgg16/vgg16.onnx')
    outputs = session.run(None, {'data': image})
    # Get top predictions
    top_predictions = get_top_k_predictions(outputs)
    print(top_predictions)

def preprocess_image(image_path, target_size=(224, 224)):
    # Read image
    img = cv2.imread(image_path)
    # Resize
    img = cv2.resize(img, target_size)
    # Convert BGR to RGB (if needed)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # Normalize (0-1 range)
    img = img.astype('float32') / 255.0
    # Mean subtraction (ImageNet standard)
    img -= [0.485, 0.456, 0.406]
    img /= [0.229, 0.224, 0.225]
    # Add batch dimension
    img = np.expand_dims(img, axis=0)
    return img

def get_top_k_predictions(outputs, k=5):
    # Load ImageNet classes
    with open('imagenet_classes.txt', 'r') as f:
        classes = [line.strip() for line in f.readlines()]
    # Get predictions
    predictions = outputs[0]
    top_k_indices = predictions.argsort()[-k:][::-1]
    # Get top k classes and probabilities
    return [(classes[idx], predictions[idx]) for idx in top_k_indices]

def batch_data_downloader_selenium(url, max_pages=9):
    # TODO figure out how to go past 100
    # either pagination or rate limit
    # might need to retrieve next page element
    destination = "data/flickr"
    driver = webdriver.Chrome()  # Make sure you have chromedriver installed
    driver.get(url)

    downloaded = 0
    last_height = driver.execute_script("return document.body.scrollHeight")

    while True:
        # Scroll down
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(2)  # Wait for images to load

        # Get all image elements
        images = driver.find_elements(By.TAG_NAME, 'img')

        # Download new images
        for img in images[downloaded:]:
            src = img.get_attribute('src')
            if src and src.startswith('http'):
                try:
                    response = requests.get(src)
                    filepath = os.path.join(destination, f'image_{downloaded}.jpg')
                    with open(filepath, 'wb') as f:
                        f.write(response.content)
                    print(f"Downloaded: {filepath}")
                    downloaded += 1
                except Exception as e:
                    print(f"Error: {e}")

        # Check if we've reached the bottom
        new_height = driver.execute_script("return document.body.scrollHeight")
        if new_height == last_height:
            break
        last_height = new_height
    driver.quit()
    return downloaded


def retrieve_cross_reference():
    # Code used to gather cross referencing data
    wildfire_url = "https://eonet.gsfc.nasa.gov/api/v3/categories/wildfires"
    response = requests.get(url=wildfire_url)
    data = response.json()
    with open('categories.json', 'w', encoding='utf-8') as f:
         json.dump(data, f, ensure_ascii=False, indent=4)

    # Example code how to query copernicus sentiel 2 data and do explcit image processing evals
    ACCESS_TOKEN = os.getenv("ACCESS_TOKEN")
    satellite = requests.post('https://sh.dataspace.copernicus.eu/api/v1/process',
    headers={f"Authorization" : "Bearer {ACCESS_TOKEN}"},
    json={
        "input": {
            "bounds": {
                "bbox": [
                    13.822174072265625,
                    45.85080395917834,
                    14.55963134765625,
                    46.29191774991382
                ]
            },
            "data": [{
                "type": "sentinel-2-l2a"
            }]
        },
        "evalscript": """
        //VERSION=3

        function setup() {
        return {
            input: ["B02", "B03", "B04"],
            output: {
            bands: 3
            }
        };
        }

        function evaluatePixel(
        sample,
        scenes,
        inputMetadata,
        customData,
        outputMetadata
        ) {
        return [2.5 * sample.B04, 2.5 * sample.B03, 2.5 * sample.B02];
        }
        """
        })

if __name__ == "__main__":
    parser  = argparse.ArgumentParser(
        prog='R&D Sandbox',
        description='Remote sensing mission utils for image classficiation and data retrieval'
        )

    # Given a flickr url, download the images using beautiful soup
    parser.add_argument('-b', '--batch-download', action='store', required=False, type=str)
    # VGG16 and ResNet50 supported for now
    parser.add_argument('-m', '--model-type', action='store', required=False, type=str)

    args = parser.parse_args()
    if args.batch_download:
        batch_data_downloader_selenium(url=args.batch_download)
    elif args.model_type:
        create_image_net_model_predictions(args.model_type)
    else:
        print("No args sent to CLI")

