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

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend

from keras_cv import bounding_box
from keras_cv.layers.preprocessing.vectorized_base_image_augmentation_layer import (
    VectorizedBaseImageAugmentationLayer,
)
from keras_cv.utils import preprocessing

# In order to support both unbatched and batched inputs, the horizontal
# and verticle axis is reverse indexed
H_AXIS = -3
W_AXIS = -2



def get_range(x, name, center=0.0):
    if isinstance(x, (float, int)):
        if x==0:
            return None
        y = [center - x, center + x]
    elif len(x) == 2 and all(
        isinstance(val, (float, int)) for val in x
    ):
        y = [x[0], x[1]]
    else:
        raise ValueError(
            f"`{name}` should be a float or "
            "a tuple or list of two floats. "
            f"Received: {x}"
        )
    return y


@keras.utils.register_keras_serializable(package="keras_cv")
class RandomAffineTransf(VectorizedBaseImageAugmentationLayer):
    """A preprocessing layer which randomly applies an affine trasformation during training.

    This layer will apply random affine trasformation to each image, filling empty space
    according to `fill_mode`.

    By default, random trasformation is only applied during training.
    At inference time, the layer does nothing. If you need to apply random
    trasformation at inference time, set `training` to True when calling the layer.

    Input pixel values can be of any range (e.g. `[0., 1.)` or `[0, 255]`) and
    of interger or floating point dtype. By default, the layer will output
    floats.

    Input shape:
      3D (unbatched) or 4D (batched) tensor with shape:
      `(..., height, width, channels)`, in `"channels_last"` format

    Output shape:
      3D (unbatched) or 4D (batched) tensor with shape:
      `(..., height, width, channels)`, in `"channels_last"` format

    Arguments:
        rotation_range: Int. Degree range for random rotations.
        zoom_range: Float or [lower, upper]. Range for random zoom. If a float,
          `[lower, upper] = [1-zoom_range, 1+zoom_range]`.
        width_shift_range: Float fraction of total width
        height_shift_range: Float fraction of total height
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
        self.zoom_range = get_range(zoom_range,'zoom_range', 1.0)
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
            theta = tf.zeros(batch_size, tf.float32)

        if self.height_shift_range:
            ty = self._random_generator.random_uniform(
                shape=[batch_size,], minval=self.height_shift_range[0], maxval=self.height_shift_range[1]
            )
        else:
            ty = tf.zeros(batch_size, tf.float32)

        if self.width_shift_range:
            tx = self._random_generator.random_uniform(
                shape=[batch_size,], minval=self.width_shift_range[0], maxval=self.width_shift_range[1]
            )
        else:
            tx = tf.zeros(batch_size, tf.float32)

        if self.shear_range:
            shear = self._random_generator.random_uniform(
                shape=[batch_size], minval=self.shear_range[0], maxval=self.shear_range[1]
            )
        else:
            shear = tf.zeros(batch_size, tf.float32)

        if self.zoom_range:
            zx = self._random_generator.random_uniform(
                shape=[batch_size], minval=self.zoom_range[0], maxval=self.zoom_range[1]
            )
            zy = self._random_generator.random_uniform(
                shape=[batch_size], minval=self.zoom_range[0], maxval=self.zoom_range[1]
            )
        else:
            zx = tf.ones(batch_size, tf.float32)
            zy = tf.ones(batch_size, tf.float32) 
        
        if self.horizontal_flip:
            zx = tf.sign(self._random_generator.random_uniform(shape=[batch_size], minval=-1, maxval=1)) * zx
        
        if self.vertical_flip:
            zy = tf.sign(self._random_generator.random_uniform(shape=[batch_size], minval=-1, maxval=1)) * zy
            
        transformations = {
            "theta": theta,
            "tx": tx,
            "ty": ty,
            "shear": shear,
            "zx": zx,
            "zy": zy,
        }

        return transformations

    def get_A(self, transformations, img_hd, img_wd):
        return get_affine_transform(
            img_hd, img_wd,
            theta=transformations["theta"],
            tx=transformations["tx"],
            ty=transformations["ty"],
            shear=transformations["shear"],
            zx=transformations["zx"],
            zy=transformations["zy"],
        )
        
    def augment_images(self, images, transformations, **kwargs):
        return self._mod_images(images, transformations)
    
    def augment_ragged_image(self, image, transformation, **kwargs):
        images = tf.expand_dims(image, axis=0)
        transformations = {k: tf.expand_dims(transformation[k], axis=0) for k in transformation}
        images = self.augment_images(images, transformations)
        return tf.squeeze(images, axis=0)
    
    def _mod_images(self, images, transformations):
        images = preprocessing.ensure_tensor(images, self.compute_dtype)
        image_shape = tf.shape(images)
        img_hd = tf.cast(image_shape[H_AXIS], tf.float32)
        img_wd = tf.cast(image_shape[W_AXIS], tf.float32)
        
        A = self.get_A(transformations, img_hd, img_wd)
        transforms = tf.stack(
            values=[
                A[..., 0, 0], A[..., 0, 1], A[..., 0, 2],
                A[..., 1, 0], A[..., 1, 1], A[..., 1, 2],
                A[..., 2, 0], A[..., 2, 1],
            ],
            axis=-1,
        )
        images = preprocessing.transform(
            images=images,
            transforms=transforms,
            fill_mode=self.fill_mode,
            fill_value=self.fill_value,
            interpolation=self.interpolation,
        )
        return images
    
    
    def augment_segmentation_masks(
        self, segmentation_masks, transformations, **kwargs
    ):
        # If segmentation_classes is specified, we have a dense segmentation mask.
        # We therefore one-hot encode before rotation to avoid bad interpolation
        # during the rotation transformation. We then make the mask sparse
        # again using tf.argmax.
        if self.segmentation_classes:
            segmentation_masks = tf.one_hot(
                tf.squeeze(segmentation_masks, axis=-1),
                self.segmentation_classes,
            )
            segmentation_masks = self._mod_images(
                segmentation_masks, transformations
            )
            segmentation_masks = tf.argmax(segmentation_masks, axis=-1)
            return tf.expand_dims(segmentation_masks, axis=-1)
        else:
            if segmentation_masks.shape[-1] == 1:
                raise ValueError(
                    "Segmentation masks must be one-hot encoded, or "
                    "RandomOperations must be initialized with "
                    "`segmentation_classes`. `segmentation_classes` was not "
                    f"specified, and mask has shape {segmentation_mask.shape}"
                )
            segmentation_masks = self._mod_images(segmentation_masks, transformations)
            # Round because we are in one-hot encoding, and we may have
            # pixels with ambugious value due to floating point math for rotation.
            return tf.round(segmentation_masks)

    def augment_bounding_boxes(
        self, bounding_boxes, transformations, images=None, **kwargs
    ):
        if self.bounding_box_format is None:
            raise ValueError(
                "`RandomOperations()` was called with bounding boxes,"
                "but no `bounding_box_format` was specified in the constructor."
                "Please specify a bounding box format in the constructor. i.e."
                "`RandomOperations(bounding_box_format='xyxy')`"
            )

        # Edge case: boxes is a tf.RaggedTensor
        if isinstance(bounding_boxes["boxes"], tf.RaggedTensor):
            bounding_boxes = bounding_box.to_dense(bounding_boxes)
        
        bounding_boxes = bounding_box.convert_format(
            bounding_boxes,
            source=self.bounding_box_format,
            target="rel_xyxy",
            images=images,
            dtype=self.compute_dtype,
        )
        
        #image_shape = tf.shape(images)
        #img_hd = tf.cast(image_shape[H_AXIS], tf.float32)
        #img_wd = tf.cast(image_shape[W_AXIS], tf.float32)
        #A = self.get_A(transformations, img_hd, img_wd)
        A = self.get_A(transformations, 1.0, 1.0)
        
        boxes = bounding_boxes["boxes"]
        ones = tf.ones_like(boxes[..., 0])
        point = tf.stack(
            [
                tf.stack([boxes[..., 0], boxes[..., 1], ones], axis=-1), 
                tf.stack([boxes[..., 2], boxes[..., 3], ones], axis=-1),
                tf.stack([boxes[..., 0], boxes[..., 3], ones], axis=-1),
                tf.stack([boxes[..., 2], boxes[..., 1], ones], axis=-1),
            ],
            axis=-1,
        )
        
        A = tf.linalg.pinv(A)
        out = tf.linalg.matmul(A[:, None, :, :], point)
        out = out[...,:2,:] / out[...,2:3,:]
        
        # find readjusted coordinates of bounding box to represent it in corners
        # format
        min_cordinates = tf.math.reduce_min(out, axis=-1)
        max_cordinates = tf.math.reduce_max(out, axis=-1)
        boxes = tf.concat([min_cordinates, max_cordinates], axis=-1)

        bounding_boxes = bounding_boxes.copy()
        bounding_boxes["boxes"] = boxes
        bounding_boxes = bounding_box.clip_to_image(
            bounding_boxes,
            bounding_box_format="rel_xyxy",
            images=images,
        )
        # cordinates cannot be float values, it is casted to int32
        bounding_boxes = bounding_box.convert_format(
            bounding_boxes,
            source="rel_xyxy",
            target=self.bounding_box_format,
            dtype=self.compute_dtype,
            images=images,
        )
        return bounding_boxes

    def augment_labels(self, labels, transformations, **kwargs):
        return labels

    
def get_translation_matrix(x, y):
    """Returns transform matrix(s) for the given translation(s).
    Returns:
      A tensor of shape `(..., 3, 3)`.
    """
    
    ones = tf.ones_like(x)
    zeros = tf.zeros_like(x)
    return tf.stack(
        values=[
            tf.stack(values=[ ones, zeros,    -x], axis = -1),
            tf.stack(values=[zeros,  ones,    -y], axis = -1),
            tf.stack(values=[zeros, zeros,  ones], axis = -1),
        ], axis=-2)

def get_rotation_matrix(theta):
    """Returns transform matrix(s) for given angle(s).
    Returns:
      A tensor of shape `(..., 3, 3)`.
    """
    theta = theta * np.pi / 180
    ones = tf.ones_like(theta)
    zeros = tf.zeros_like(theta)
    cos = tf.math.cos(theta)
    sin = tf.math.sin(theta)
    return tf.stack(
        values=[
            tf.stack(values=[  cos,  -sin, zeros], axis = -1),
            tf.stack(values=[  sin,   cos, zeros], axis = -1),
            tf.stack(values=[zeros, zeros,  ones], axis = -1),
        ], axis=-2)

def get_shear_matrix(shear):
    """Build ransform matrix(s) for given shear(s).
    Returns:
      A tensor of shape `(..., 3, 3)`
    """
    shear = shear * np.pi / 180
    ones = tf.ones_like(shear)
    zeros = tf.zeros_like(shear)
    cos = tf.math.cos(shear)
    sin = tf.math.sin(shear)
    return tf.stack(
        values=[
            tf.stack(values=[ ones,  -sin, zeros], axis = -1),
            tf.stack(values=[zeros,   cos, zeros], axis = -1),
            tf.stack(values=[zeros, zeros,  ones], axis = -1),
        ], axis=-2)

def get_zoom_matrix(zx, zy):
    """Build transform matrix(s) for given zoom.
    Returns:
      A tensor of shape `(..., 3, 3)`
    """
    ones = tf.ones_like(zx)
    zeros = tf.zeros_like(zx)
    return tf.stack(
        values=[
            tf.stack(values=[1./zx, zeros, zeros], axis = -1),
            tf.stack(values=[zeros, 1./zy, zeros], axis = -1),
            tf.stack(values=[zeros, zeros,  ones], axis = -1),
        ], axis=-2)

def get_affine_transform(
    img_hd, img_wd,
    theta,
    tx,
    ty,
    shear,
    zx,
    zy,
    name=None
):
    with backend.name_scope(name or "translation_matrix"):
        o_x = img_wd / 2.0
        o_y = img_hd / 2.0
        
        transform_matrix = get_translation_matrix(-o_x, -o_y)[None, ...]
        transform_matrix = tf.linalg.matmul(transform_matrix, get_rotation_matrix(theta))
        transform_matrix = tf.linalg.matmul(transform_matrix, get_translation_matrix(img_wd*tx, img_hd*ty))
        transform_matrix = tf.linalg.matmul(transform_matrix, get_shear_matrix(shear))
        transform_matrix = tf.linalg.matmul(transform_matrix, get_zoom_matrix(zx, zy))
        transform_matrix = tf.linalg.matmul(transform_matrix, get_translation_matrix(o_x, o_y)[None, ...])
        
        return transform_matrix


