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

import tensorflow as tf
# import cv2 as cv
import os
import requests
import json
# import pandas
import onnxruntime as backend

import numpy as np
import onnx
import os
import glob


def onnx_vgg_model_test():
    model = onnx.load('vgg16/vgg16.onnx')
    test_data_dir = 'vgg16/test_data_set_0'

    # Load inputs
    inputs = []
    inputs_num = len(glob.glob(os.path.join(test_data_dir, 'input_*.pb')))
    for i in range(inputs_num):
        input_file = os.path.join(test_data_dir, 'input_{}.pb'.format(i))
        tensor = onnx.TensorProto()
        with open(input_file, 'rb') as f:
            tensor.ParseFromString(f.read())
        inputs.append(numpy_helper.to_array(tensor))

    # Load reference outputs
    ref_outputs = []
    ref_outputs_num = len(glob.glob(os.path.join(test_data_dir, 'output_*.pb')))
    for i in range(ref_outputs_num):
        output_file = os.path.join(test_data_dir, 'output_{}.pb'.format(i))
        tensor = onnx.TensorProto()
        with open(output_file, 'rb') as f:
            tensor.ParseFromString(f.read())
        ref_outputs.append(numpy_helper.to_array(tensor))

    # Run the model on the backend
    outputs = list(tf.run_model(model, inputs))

    # Compare the results with reference outputs.
    for ref_o, o in zip(ref_outputs, outputs):
        np.testing.assert_almost_equal(ref_o, o)

def retrieve():
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
    onnx_vgg_model_test()
    # retrieve()
