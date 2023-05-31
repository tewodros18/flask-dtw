from flask import *
import numpy as np
import pandas as pd
from fastdtw import fastdtw
from sign_recorder import SignRecorder
from utils.dataset_utils import load_reference_signs
app = Flask(__name__)

videos = ['ha-1', 'ha-2', 'Hu-1']
# Create a DataFrame of reference signs (name: str, model: SignModel, distance: int)
reference_signs = load_reference_signs(videos)
# Object that stores mediapipe results and computes sign similarities
sign_recorder = SignRecorder(reference_signs)


@app.route('/')
def hello_world():
    return 'Hello, World!'

@app.route('/dtw', methods=["POST"])
def dtw():
    #x = np.array([[1,1], [2,2], [3,3], [4,4], [5,5]])
    x = request.json["list1"]
    #y = np.array([[2,2], [3,3], [4,4]])
    y = request.json["list2"]
    distance, path = fastdtw(x, y)
    return str(distance)

@app.route('/process', methods=["POST"])
def get_prediction():
    result = sign_recorder.process_results(request.json["results"])
    return result 