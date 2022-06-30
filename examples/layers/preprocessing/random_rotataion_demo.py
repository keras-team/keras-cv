# Copyright 2022 The KerasCV Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
   random_rotation_demo.py shows how to use the RandomRotation preprocessing layer
   for object detection. An image is downloaded from URL, both the image and 
   bounding boxes are augmented and displayed using matplotlib.
"""
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import requests
import tensorflow as tf
from PIL import Image
from keras_cv.layers import preprocessing

IMG_SIZE = (256, 256)
BATCH_SIZE = 9


def visualize_bounding_boxes_on_image(image, bboxes, color="r", title="visualization"):
    fig, ax = plt.subplots()
    ax.imshow(image)
    for box in bboxes:
        rect = patches.Rectangle(
            (box[0], box[1]),
            box[2] - box[0],
            box[3] - box[1],
            linewidth=1,
            edgecolor=color,
            facecolor="none",
        )
        ax.add_patch(rect)
    plt.title(title)
    plt.show()


def main():
    image = Image.open(
        requests.get("http://www.lenna.org/lena_std.tif", stream=True).raw
    )
    bboxes = tf.convert_to_tensor([[200, 200, 400, 400], [100, 100, 300, 300]])
    visualize_bounding_boxes_on_image(image, bboxes, title="Before Augmentation")
    randomrotation = preprocessing.RandomRotation(
        factor=(0.5, 0.5),
        bounding_box_format="xyxy",
    )
    input = {"images": image, "bounding_boxes": bboxes}
    result = randomrotation(input)
    out_images, out_bboxes = result["images"], result["bounding_boxes"]
    visualize_bounding_boxes_on_image(
        out_images.numpy().astype("uint8"), out_bboxes, title="After Augmentation"
    )


if __name__ == "__main__":
    main()
