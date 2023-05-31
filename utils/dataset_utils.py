import os
import pandas as pd
from models.sign_model import SignModel
from utils.landmark_utils import load_array


def load_reference_signs(videos):
    reference_signs = {"name": [], "sign_model": [], "distance": []}
    print(videos)
    for video_name in videos:
        sign_name = video_name.split("-")[0]
        path = os.path.join("data", "dataset", sign_name, video_name)

        left_hand_list = load_array(os.path.join(path, f"lh_{video_name}.pickle"))
        right_hand_list = load_array(os.path.join(path, f"rh_{video_name}.pickle"))

        reference_signs["name"].append(sign_name)
        reference_signs["sign_model"].append(SignModel(left_hand_list, right_hand_list))
        reference_signs["distance"].append(0)
    out_file = open("myfile.json", "w")
    reference_signs = pd.DataFrame(reference_signs, dtype=object)
    print(reference_signs)
    print(
        f'Dictionary count: {reference_signs[["name", "sign_model"]].groupby(["name"]).count()}'
    )
    return reference_signs
