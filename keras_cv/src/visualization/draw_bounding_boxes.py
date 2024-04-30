# Copyright 2023 The KerasCV Authors
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

try:
    import cv2
except:
    cv2 = None

import numpy as np

from keras_cv.src import bounding_box
from keras_cv.src import utils
from keras_cv.src.api_export import keras_cv_export
from keras_cv.src.utils import assert_cv2_installed


@keras_cv_export("keras_cv.visualization.draw_bounding_boxes")
def draw_bounding_boxes(
    images,
    bounding_boxes,
    color,
    bounding_box_format,
    line_thickness=1,
    text_thickness=1,
    font_scale=1.0,
    class_mapping=None,
):
    """Internal utility to draw bounding boxes on the target image.

    Accepts a batch of images and batch of bounding boxes. The function draws
    the bounding boxes onto the image, and returns a new image tensor with the
    annotated images. This API is intentionally not exported, and is considered
    an implementation detail.

    Args:
        images: a batch Tensor of images to plot bounding boxes onto.
        bounding_boxes: a Tensor of batched bounding boxes to plot onto the
            provided images.
        color: the color in which to plot the bounding boxes
        bounding_box_format: The format of bounding boxes to plot onto the
            images. Refer
            [to the keras.io docs](https://keras.io/api/keras_cv/bounding_box/formats/)
            for more details on supported bounding box formats.
        line_thickness: (Optional) line_thickness for the box and text labels.
            Defaults to 2.
        text_thickness: (Optional) the thickness for the text, defaults to
            `1.0`.
        font_scale: (Optional) scale of font to draw in, defaults to `1.0`.
        class_mapping: (Optional) dictionary from class ID to class label.

    Returns:
        the input `images` with provided bounding boxes plotted on top of them
    """  # noqa: E501
    assert_cv2_installed("draw_bounding_boxes")
    bounding_boxes = bounding_box.convert_format(
        bounding_boxes, source=bounding_box_format, target="xyxy", images=images
    )
    text_thickness = text_thickness or line_thickness

    bounding_boxes["boxes"] = utils.to_numpy(bounding_boxes["boxes"])
    bounding_boxes["classes"] = utils.to_numpy(bounding_boxes["classes"])
    images = utils.to_numpy(images)
    image_width = images.shape[-2]
    outline_factor = image_width // 100

    class_mapping = class_mapping or {}
    result = []

    if len(images.shape) != 4:
        raise ValueError(
            "Images must be a batched np-like with elements of shape "
            "(height, width, 3)"
        )

    for i in range(images.shape[0]):
        bounding_box_batch = {
            "boxes": bounding_boxes["boxes"][i],
            "classes": bounding_boxes["classes"][i],
        }
        if "confidence" in bounding_boxes:
            bounding_box_batch["confidence"] = bounding_boxes["confidence"][i]

        image = utils.to_numpy(images[i]).astype("uint8")
        for b_id in range(bounding_box_batch["boxes"].shape[0]):
            x, y, x2, y2 = bounding_box_batch["boxes"][b_id].astype(int)
            class_id = bounding_box_batch["classes"][b_id].astype(int)
            confidence = bounding_box_batch.get("confidence", None)

            if class_id == -1:
                continue
            # force conversion back to contiguous array
            x, y, x2, y2 = int(x), int(y), int(x2), int(y2)
            cv2.rectangle(
                image,
                (x, y),
                (x2, y2),
                (0, 0, 0, 0.5),
                line_thickness + outline_factor,
            )
            cv2.rectangle(image, (x, y), (x2, y2), color, line_thickness)
            class_id = int(class_id)

            if class_id in class_mapping:
                label = class_mapping[class_id]
                if confidence is not None:
                    label = f"{label} | {confidence[b_id]:.2f}"

                x, y = _find_text_location(
                    x, y, font_scale, line_thickness, outline_factor
                )
                cv2.putText(
                    image,
                    label,
                    (x, y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale,
                    (0, 0, 0, 0.5),
                    text_thickness + outline_factor,
                )
                cv2.putText(
                    image,
                    label,
                    (x, y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale,
                    color,
                    text_thickness,
                )
        result.append(image)
    return np.array(result).astype(int)


def _find_text_location(x, y, font_scale, line_thickness, outline_factor):
    font_height = int(font_scale * 12)
    target_y = y - int(8 + outline_factor)
    if target_y - (2 * font_height) > 0:
        return x, y - int(8 + outline_factor)

    line_offset = line_thickness + outline_factor
    static_offset = 3

    return (
        x + outline_factor + static_offset,
        y + (2 * font_height) + line_offset + static_offset,
    )
