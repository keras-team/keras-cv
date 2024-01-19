import json
from pathlib import Path

import cv2
import keras
import numpy as np
from label_generator import generate_text_probability_map
from label_generator import generate_threshold_label


def read_label_file(file_path):
    with open(file_path, "r") as f:
        data = f.readlines()
    return data


def decode_image(image_path):
    try:
        image = cv2.imread(image_path)
        if image.shape[-1] == 1:
            image = np.dstack([image, image, image])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image
    except:
        return None


def pad_polygons(polygons):
    max_num_points = 0
    for polygon in polygons:
        if len(polygon) > max_num_points:
            max_num_points = len(polygon)
    padded_polygons = list()
    for polygon in polygons:
        polygon = polygon + [polygon[-1]] * (max_num_points - len(polygon))
        padded_polygons.append(polygon)
    return padded_polygons


def decode_label(label):
    label = json.loads(label)
    polygons, texts, ignore_flags = list(), list(), list()
    for info in label:
        text = info["transcription"]
        if text in ["*", "###"]:
            ignore_flags.append(True)
        else:
            ignore_flags.append(False)
        polygon = info["points"]
        polygons.append(polygon)
        texts.append(text)
    polygons = pad_polygons(polygons)
    return polygons, texts, ignore_flags


class ICDARPyDataset(keras.utils.PyDataset):
    def __init__(self, image_dir, file_path, **kwargs):
        super().__init__(**kwargs)
        self.image_dir = image_dir
        self.file_path = file_path
        self.data_lines = read_label_file(file_path)

    def __len__(self):
        return len(self.data_lines)

    def __getitem__(self, idx):
        # 1. Grab a single line from the data_lines list
        line = self.data_lines[idx]
        # 2. Split the line into image_path and label
        image_path, label = line.strip().split("\t")
        image_path = str(Path(self.image_dir) / Path(image_path).name)
        # 3. Decode the image
        image = decode_image(image_path)
        if image is None:
            return dict()
        # 4. Decode the labels
        polygons, texts, ignore_flags = decode_label(label)
        if len(polygons) == 0:
            return dict()
        else:
            data = {
                "image": image,
                "polygons": np.array(polygons),
                "texts": np.array(texts),
                "ignore_flags": np.array(ignore_flags),
            }
            generate_text_probability_map(data)
            generate_threshold_label(data)
            return data
