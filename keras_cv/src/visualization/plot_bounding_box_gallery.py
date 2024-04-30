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

import functools

import numpy as np

from keras_cv.src import bounding_box
from keras_cv.src import utils
from keras_cv.src.utils import assert_cv2_installed
from keras_cv.src.utils import assert_matplotlib_installed
from keras_cv.src.visualization.draw_bounding_boxes import draw_bounding_boxes
from keras_cv.src.visualization.plot_image_gallery import plot_image_gallery

try:
    from matplotlib import patches
except:
    patches = None

from keras_cv.src.api_export import keras_cv_export


@keras_cv_export("keras_cv.visualization.plot_bounding_box_gallery")
def plot_bounding_box_gallery(
    images,
    value_range,
    bounding_box_format,
    y_true=None,
    y_pred=None,
    true_color=(0, 188, 212),
    pred_color=(255, 235, 59),
    line_thickness=2,
    font_scale=1.0,
    text_thickness=None,
    class_mapping=None,
    ground_truth_mapping=None,
    prediction_mapping=None,
    legend=False,
    legend_handles=None,
    rows=3,
    cols=3,
    **kwargs
):
    """Plots a gallery of images with corresponding bounding box annotations.

    Example:
    ```python
    train_ds = tfds.load(
        "voc/2007", split="train", with_info=False, shuffle_files=True
    )

    def unpackage_tfds_inputs(inputs):
        image = inputs["image"]
        boxes = inputs["objects"]["bbox"]
        bounding_boxes = {"classes": classes, "boxes": boxes}
        return image, bounding_boxes

    train_ds = train_ds.map(unpackage_tfds_inputs)
    train_ds = train_ds.apply(tf.data.experimental.dense_to_ragged_batch(16))
    images, boxes = next(iter(train_ds.take(1)))

    keras_cv.visualization.plot_bounding_box_gallery(
        images,
        value_range=(0, 255),
        bounding_box_format="xywh",
        y_true=boxes,
        scale=3,
        rows=2,
        cols=2,
        line_thickness=4,
        font_scale=1,
        legend=True,
    )
    ```

    ![Example bounding box gallery](https://i.imgur.com/tJpb8hZ.png)

    Args:
        images: a Tensor or NumPy array containing images to show in the
            gallery.
        value_range: value range of the images. Common examples include
            `(0, 255)` and `(0, 1)`.
        bounding_box_format: the bounding_box_format the provided bounding boxes
            are in.
        y_true: (Optional) a KerasCV bounding box dictionary representing the
            ground truth bounding boxes.
        y_pred: (Optional) a KerasCV bounding box dictionary representing the
            predicted bounding boxes.
        pred_color: three element tuple representing the color to use for
            plotting predicted bounding boxes.
        true_color: three element tuple representing the color to use for
            plotting true bounding boxes.
        class_mapping: (Optional) class mapping from class IDs to strings
        ground_truth_mapping: (Optional) class mapping from class IDs to
            strings, defaults to `class_mapping`
        prediction_mapping: (Optional) class mapping from class IDs to strings,
            defaults to `class_mapping`
        line_thickness: (Optional) line_thickness for the box and text labels.
            Defaults to 2.
        text_thickness: (Optional) the line_thickness for the text, defaults to
            `1.0`.
        font_scale: (Optional) font size to draw bounding boxes in.
        legend: whether to create a legend with the specified colors for
            `y_true` and `y_pred`, defaults to False.
        kwargs: keyword arguments to propagate to
            `keras_cv.visualization.plot_image_gallery()`.
    """
    assert_matplotlib_installed("plot_bounding_box_gallery")
    assert_cv2_installed("plot_bounding_box_gallery")

    prediction_mapping = prediction_mapping or class_mapping
    ground_truth_mapping = ground_truth_mapping or class_mapping

    plotted_images = utils.to_numpy(images)

    draw_fn = functools.partial(
        draw_bounding_boxes,
        bounding_box_format="xyxy",
        line_thickness=line_thickness,
        text_thickness=text_thickness,
        font_scale=font_scale,
    )

    if y_true is not None:
        y_true = y_true.copy()
        y_true["boxes"] = utils.to_numpy(y_true["boxes"])
        y_true["classes"] = utils.to_numpy(y_true["classes"])
        y_true = bounding_box.convert_format(
            y_true, images=images, source=bounding_box_format, target="xyxy"
        )
        plotted_images = draw_fn(
            plotted_images,
            y_true,
            true_color,
            class_mapping=ground_truth_mapping,
        )

    if y_pred is not None:
        y_pred = y_pred.copy()
        y_pred["boxes"] = utils.to_numpy(y_pred["boxes"])
        y_pred["classes"] = utils.to_numpy(y_pred["classes"])
        y_pred = bounding_box.convert_format(
            y_pred, images=images, source=bounding_box_format, target="xyxy"
        )
        plotted_images = draw_fn(
            plotted_images, y_pred, pred_color, class_mapping=prediction_mapping
        )

    if legend:
        if legend_handles:
            raise ValueError(
                "Only pass `legend` OR `legend_handles` to "
                "`luketils.visualization.plot_bounding_box_gallery()`."
            )
        legend_handles = [
            patches.Patch(
                color=np.array(true_color) / 255.0,
                label="Ground Truth",
            ),
            patches.Patch(
                color=np.array(pred_color) / 255.0,
                label="Prediction",
            ),
        ]

    return plot_image_gallery(
        plotted_images,
        value_range,
        legend_handles=legend_handles,
        rows=rows,
        cols=cols,
        **kwargs
    )
