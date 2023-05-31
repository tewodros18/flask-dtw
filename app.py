from flask import Flask
import numpy as np
import pandas as pd
from fastdtw import fastdtw
app = Flask(__name__)

@app.route('/')
def hello_world():
    return 'Hello, World!'

@app.route('/dtw')
def dtw():
    x = np.array([[1,1], [2,2], [3,3], [4,4], [5,5]])
    y = np.array([[2,2], [3,3], [4,4]])
    distance, path = fastdtw(x, y)
    return distance
