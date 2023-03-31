# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================


"""Utilies for image preprocessing and augmentation.

Deprecated: `tf.keras.preprocessing.image` APIs do not operate on tensors and
are not recommended for new code. Prefer loading data with
`tf.keras.utils.image_dataset_from_directory`, and then transforming the output
`tf.data.Dataset` with preprocessing layers. For more information, see the
tutorials for [loading images](
https://www.tensorflow.org/tutorials/load_data/images) and [augmenting images](
https://www.tensorflow.org/tutorials/images/data_augmentation), as well as the
[preprocessing layer guide](
https://www.tensorflow.org/guide/keras/preprocessing_layers).
"""

import collections
import multiprocessing
import os
import threading
import warnings

import numpy as np

from keras_cv.layers.preprocessing.base_image_augmentation_layer import BaseImageAugmentationLayer
from keras_cv.utils import preprocessing
H_AXIS = -3
W_AXIS = -2



def get_range(x, name, center=0.0):
    if isinstance(x, (float, int)):
        if x==0:
            return None
        y = [center - x, center + x]
    elif len(zoom_range) == 2 and all(
        isinstance(val, (float, int)) for val in zoom_range
    ):
        y = [zoom_range[0], zoom_range[1]]
    else:
        raise ValueError(
            "`%s` should be a float or " % name
            "a tuple or list of two floats. "
            "Received: %s" % (zoom_range,)
        )
    return y

class RandomAffineTransf(BaseImageAugmentationLayer):
    """Generate batches of tensor image data with real-time data augmentation.

     The data will be looped over (in batches).

    Args:
        rotation_range: Int. Degree range for random rotations.
        zoom_range: Float or [lower, upper]. Range for random zoom. If a float,
          `[lower, upper] = [1-zoom_range, 1+zoom_range]`.
        width_shift_range: Float, 1-D array-like float fraction of total width
        height_shift_range: Float, 1-D array-like float fraction of total height
        shear_range: Float. Shear Intensity (Shear angle in counter-clockwise
          direction in degrees)
        horizontal_flip: Boolean. Randomly flip inputs horizontally.
        vertical_flip: Boolean. Randomly flip inputs vertically.
        fill_mode: Points outside the boundaries of the input are filled according
          to the given mode (one of `{"constant", "reflect", "wrap", "nearest"}`).
          - *reflect*: `(d c b a | a b c d | d c b a)` The input is extended by
            reflecting about the edge of the last pixel.
          - *constant*: `(k k k k | a b c d | k k k k)` The input is extended by
            filling all values beyond the edge with the same constant value k = 0.
          - *wrap*: `(a b c d | a b c d | a b c d)` The input is extended by
            wrapping around to the opposite edge.
          - *nearest*: `(a a a a | a b c d | d d d d)` The input is extended by
            the nearest pixel.
        interpolation: Interpolation mode. Supported values: `"nearest"`,
          `"bilinear"`.
        seed: Integer. Used to create a random seed.
        fill_value: a float represents the value to be filled outside the
          boundaries when `fill_mode="constant"`.
        bounding_box_format: The format of bounding boxes of input dataset. Refer
          https://github.com/keras-team/keras-cv/blob/master/keras_cv/bounding_box/converters.py
          for more details on supported bounding box formats.
        segmentation_classes: an optional integer with the number of classes in
          the input segmentation mask. Required iff augmenting data with sparse
          (non one-hot) segmentation masks. Include the background class in this
          count (e.g. for segmenting dog vs background, this should be set to 2).
          
    """

    def __init__(
        self,
        rotation_range=0.0,
        zoom_range=0.0,
        width_shift_range=0.0,
        height_shift_range=0.0,
        shear_range=0.0,
        horizontal_flip=False,
        vertical_flip=False,
        fill_mode="reflect",
        interpolation="bilinear",
        seed=None,
        fill_value=0.0,
        bounding_box_format=None,
        segmentation_classes=None,
        **kwargs,
    ):
        super().__init__(seed=seed, force_generator=True, **kwargs)
        
        self.rotation_range = get_range(rotation_range,'rotation_range')
        self.zoom_range = get_range(zoom_range,'rotation_range', 1.0)
        self.width_shift_range = get_range(width_shift_range,'width_shift_range')
        self.height_shift_range = get_range(height_shift_range,'height_shift_range')
        self.shear_range = get_range(shear_range,'shear_range')
        self.horizontal_flip = horizontal_flip
        self.vertical_flip = vertical_flip

        self.fill_mode = fill_mode
        self.fill_value = fill_value
        self.interpolation = interpolation
        self.seed = seed
        self.bounding_box_format = bounding_box_format
        self.segmentation_classes = segmentation_classes
    
    def get_config(self):
        config = {
            "rotation_range":self.rotation_range,
            "zoom_range":self.zoom_range,
            "width_shift_range":self.width_shift_range,
            "height_shift_range":self.height_shift_range,
            "shear_range":self.shear_range,
            "horizontal_flip":self.horizontal_flip,
            "vertical_flip":self.vertical_flip,
            "fill_mode":self.fill_mode,
            "fill_value":self.fill_value,
            "interpolation":self.interpolation,
            "seed":self.seed,
            "bounding_box_format":self.bounding_box_format,
            "segmentation_classes":self.segmentation_classes,
        }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))

    
    def get_random_transformation_batch(self, batch_size, **kwargs):
        """Generates random parameters for a transformation.

        Returns:
            A dictionary containing randomly chosen parameters describing the
            transformation.
        """
        

        if self.rotation_range:
            theta = self._random_generator.random_uniform(
                shape=[batch_size,], minval=self.rotation_range[0], maxval=self.rotation_range[1]
            )
        else:
            theta = tf.cast([0,]*batch_size, tf.float32)

        if self.height_shift_range:
            tx = self._random_generator.random_uniform(
                shape=[batch_size,], minval=-self.height_shift_range[0], maxval=self.height_shift_range[1]
            )
        else:
            tx = tf.cast([0,]*batch_size, tf.float32)

        if self.width_shift_range:
            ty = self._random_generator.random_uniform(
                shape=[batch_size,], minval=-self.width_shift_range[0], maxval=self.width_shift_range[1]
            )
        else:
            ty = tf.cast([0,]*batch_size, tf.float32)

        if self.shear_range:
            shear = self._random_generator.random_uniform(
                shape=[batch_size], minval=self.shear_range[0], maxval=self.shear_range[1]
            )
        else:
            shear = tf.cast([0,]*batch_size, tf.float32)

        if self.zoom_range:
            zx = self._random_generator.random_uniform(
                shape=[batch_size], minval=self.zoom_range[0], maxval=self.zoom_range[1]
            )
            zy = self._random_generator.random_uniform(
                shape=[batch_size], minval=self.zoom_range[0], maxval=self.zoom_range[1]
            )
        else:
            zx = tf.cast([1,]*batch_size, tf.float32)
            zy = tf.cast([1,]*batch_size, tf.float32) 
        
        if self.horizontal_flip:
            zx = torch.sign(self._random_generator.random_uniform(shape=[batch_size], minval=-1, maxval=1)) * zx
        if self.vertical_flip:
            zy = torch.sign(self._random_generator.random_uniform(shape=[batch_size], minval=-1, maxval=1)) * zy
            
        transform_parameters = {
            "theta": theta,
            "tx": tx,
            "ty": ty,
            "shear": shear,
            "zx": zx,
            "zy": zy,
        }

        return transform_parameters

    def get_A(transform_parameters, img_hd, img_wd):
        return tf.stack([tf.cast(get_affine_transform(
            img_hd, img_wd,
            theta=transform_parameters["theta"][i],
            tx=img_wd*transform_parameters["tx"][i],
            ty=img_hd*transform_parameters["ty"][i],
            shear=transform_parameters["shear"][i],
            zx=transform_parameters["zx"][i],
            zy=transform_parameters["zy"][i],
        ), tf.float32) for i in range(len(transform_parameters["zy"]))], 0)
        
    def augment_image(self, image, transformation, **kwargs):
        return self._rotate_image(image, transformation)
        
    def _mod_image(self, image, transformation):
        image = preprocessing.ensure_tensor(image, self.compute_dtype)
        original_shape = image.shape
        image = tf.expand_dims(image, 0)
        image_shape = tf.shape(image)
        img_hd = tf.cast(image_shape[H_AXIS], tf.float32)
        img_wd = tf.cast(image_shape[W_AXIS], tf.float32)
        
        A = get_A(transform_parameters, img_hd, img_wd)
        A = tf.concat(
            values=[
                A[:, 0, 0, None],
                A[:, 0, 1, None],
                A[:, 0, 2, None],
                A[:, 1, 0, None],
                A[:, 1, 1, None],
                A[:, 1, 2, None],
                tf.zeros((len(A), 2), tf.float32),
            ],
            axis=1,
        )
        output = preprocessing.transform(
            image,
            A,
            fill_mode=self.fill_mode,
            fill_value=self.fill_value,
            interpolation=self.interpolation,
        )
        output = tf.squeeze(output, 0)
        output.set_shape(original_shape)
        return output
    
    
    def augment_segmentation_mask(
        self, segmentation_mask, transformation, **kwargs
    ):
        # If segmentation_classes is specified, we have a dense segmentation mask.
        # We therefore one-hot encode before rotation to avoid bad interpolation
        # during the rotation transformation. We then make the mask sparse
        # again using tf.argmax.
        if self.segmentation_classes:
            one_hot_mask = tf.one_hot(
                tf.squeeze(segmentation_mask, axis=-1),
                self.segmentation_classes,
            )
            rotated_one_hot_mask = self._mod_image(
                one_hot_mask, transformation
            )
            rotated_mask = tf.argmax(rotated_one_hot_mask, axis=-1)
            return tf.expand_dims(rotated_mask, axis=-1)
        else:
            if segmentation_mask.shape[-1] == 1:
                raise ValueError(
                    "Segmentation masks must be one-hot encoded, or "
                    "RandomOperations must be initialized with "
                    "`segmentation_classes`. `segmentation_classes` was not "
                    f"specified, and mask has shape {segmentation_mask.shape}"
                )
            rotated_mask = self._mod_image(segmentation_mask, transformation)
            # Round because we are in one-hot encoding, and we may have
            # pixels with ambugious value due to floating point math for rotation.
            return tf.round(rotated_mask)

    def augment_bounding_boxes(
        self, bounding_boxes, transformation, image=None, **kwargs
    ):
        if self.bounding_box_format is None:
            raise ValueError(
                "`RandomOperations()` was called with bounding boxes,"
                "but no `bounding_box_format` was specified in the constructor."
                "Please specify a bounding box format in the constructor. i.e."
                "`RandomOperations(bounding_box_format='xyxy')`"
            )

        bounding_boxes = bounding_box.convert_format(
            bounding_boxes,
            source=self.bounding_box_format,
            target="xyxy",
            images=image,
        )
        image_shape = tf.shape(image)
        img_hd = tf.cast(image_shape[H_AXIS], tf.float32)
        img_wd = tf.cast(image_shape[W_AXIS], tf.float32)
        A = get_A(transform_parameters, img_hd, img_wd).T
        
        boxes = bounding_boxes["boxes"]
        point = tf.stack(
            [
                tf.stack([boxes[:, 0], boxes[:, 1], 1+0*boxes[:, 0]], axis=1),
                tf.stack([boxes[:, 2], boxes[:, 1], 1+0*boxes[:, 0]], axis=1),
                tf.stack([boxes[:, 2], boxes[:, 3], 1+0*boxes[:, 0]], axis=1),
                tf.stack([boxes[:, 0], boxes[:, 3], 1+0*boxes[:, 0]], axis=1),
            ],
            axis=1,
        )
        out = point @ A[None,:,:]
        out = out[:,:,:2]
        
        # find readjusted coordinates of bounding box to represent it in corners
        # format
        min_cordinates = tf.math.reduce_min(out, axis=1)
        max_cordinates = tf.math.reduce_max(out, axis=1)
        boxes = tf.concat([min_cordinates, max_cordinates], axis=1)

        bounding_boxes = bounding_boxes.copy()
        bounding_boxes["boxes"] = boxes
        bounding_boxes = bounding_box.clip_to_image(
            bounding_boxes,
            bounding_box_format="xyxy",
            images=image,
        )
        # cordinates cannot be float values, it is casted to int32
        bounding_boxes = bounding_box.convert_format(
            bounding_boxes,
            source="xyxy",
            target=self.bounding_box_format,
            dtype=self.compute_dtype,
            images=image,
        )
        return bounding_boxes

    def augment_label(self, label, transformation, **kwargs):
        return label

def transform_matrix_offset_center(matrix, x, y):
    o_x = float(x) / 2 - 0.5
    o_y = float(y) / 2 - 0.5
    offset_matrix = np.array([[1, 0, o_x], [0, 1, o_y], [0, 0, 1]])
    reset_matrix = np.array([[1, 0, -o_x], [0, 1, -o_y], [0, 0, 1]])
    transform_matrix = np.dot(np.dot(offset_matrix, matrix), reset_matrix)
    return transform_matrix


def get_affine_transform(
    img_hd, img_wd,
    theta=0,
    tx=0,
    ty=0,
    shear=0,
    zx=1,
    zy=1,
):
    
    transform_matrix = None
    if theta != 0:
        theta = np.deg2rad(theta)
        rotation_matrix = np.array(
            [
                [np.cos(theta), -np.sin(theta), 0],
                [np.sin(theta), np.cos(theta), 0],
                [0, 0, 1],
            ]
        )
        transform_matrix = rotation_matrix

    if tx != 0 or ty != 0:
        shift_matrix = np.array([[1, 0, tx], [0, 1, ty], [0, 0, 1]])
        if transform_matrix is None:
            transform_matrix = shift_matrix
        else:
            transform_matrix = np.dot(transform_matrix, shift_matrix)

    if shear != 0:
        shear = np.deg2rad(shear)
        shear_matrix = np.array(
            [[1, -np.sin(shear), 0], [0, np.cos(shear), 0], [0, 0, 1]]
        )
        if transform_matrix is None:
            transform_matrix = shear_matrix
        else:
            transform_matrix = np.dot(transform_matrix, shear_matrix)

    if zx != 1 or zy != 1:
        zoom_matrix = np.array([[zx, 0, 0], [0, zy, 0], [0, 0, 1]])
        if transform_matrix is None:
            transform_matrix = zoom_matrix
        else:
            transform_matrix = np.dot(transform_matrix, zoom_matrix)
    
    transform_matrix = transform_matrix_offset_center(
        transform_matrix, img_hd, img_wd,
    )
       
    return transform_matrix


