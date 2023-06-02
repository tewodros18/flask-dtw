from flask import *
import numpy as np
import pandas as pd
from fastdtw import fastdtw
from sign_recorder import SignRecorder
from utils.dataset_utils import load_reference_signs
app = Flask(__name__)

videos = ['he-1', 'he-2', 'he-3', 'he-4', 'he-5', 'he-6', 'hu-1', 'hu-2', 'hu-3', 'hu-4', 'hu-5', 'hu-6', 'Ne-1', 'Ne-2', 'Ne-3', 'Ne-4', 'Ne-5']
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