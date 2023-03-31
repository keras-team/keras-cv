# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
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
"""Keras image dataset loading utilities."""
""" modified by davide """

import os
import pandas
import numpy as np
import tensorflow.compat.v2 as tf

from keras.utils import dataset_utils
from keras.utils import image_utils

from keras_cv.layers.preprocessing.base_image_augmentation_layer import IMAGES
from keras_cv.layers.preprocessing.base_image_augmentation_layer import LABELS
from keras_cv.layers.preprocessing.base_image_augmentation_layer import BOUNDING_BOXES 
from keras_cv.layers.preprocessing.base_image_augmentation_layer import KEYPOINTS 
from keras_cv.layers.preprocessing.base_image_augmentation_layer import SEGMENTATION_MASKS

ALLOWLIST_FORMATS = (".bmp", ".gif", ".jpeg", ".jpg", ".png")

def dataframe_from_directory(
    directory,
    formats=ALLOWLIST_FORMATS,
    follow_links=False,
):
    """Generates a `pandas.DataFrame` from image files in a directory.

    If your directory structure is:

    ```
    main_directory/
    ...class_a/
    ......a_image_1.jpg
    ......a_image_2.jpg
    ...class_b/
    ......b_image_1.jpg
    ......b_image_2.jpg
    ```

    Then calling `image_dataframe_from_directory(main_directory)` will 
    return a `pandas.DataFrame` that yields batches of
    images from the subdirectories `class_a` and `class_b`.

    Supported image formats: jpeg, png, bmp, gif.
    Animated gifs are truncated to the first frame.

    Args:
      directory: Directory where the data is located.
      follow_links: Whether to visit subdirectories pointed to by symlinks.
          Defaults to False.

    Returns:
      A `pandas.DataFrame` object.

    """    
    image_paths, labels, class_names = dataset_utils.index_directory(
        directory,
        labels='inferred',
        formats=formats,
        shuffle=False,
        follow_links=follow_links,
    )
    dataframe = pandas.DataFrame({'image': image_paths,
                                  'class': [class_names[_] for _ in labels]})
    return dataframe

     
def load_img(
    path, num_channels, image_size=None, interpolation=None
):
    """Load an image from a path and resize it."""
    img = tf.io.read_file(path)
    img = tf.image.decode_image(
        img, channels=num_channels, expand_animations=False
    )
    if image_size:
        img = image_utils.smart_resize(
            img, image_size, interpolation=interpolation
        )
        img.set_shape((image_size[0], image_size[1], num_channels))
    return img


def _get_fun_load_class(class_mode, class_names=None):
    if (not isinstance(class_mode, str)) and  callable(class_mode):
        if class_names is not None:
            raise ValueError(
                'You can only pass `class_names` if '
                '`class_mode` is "int" or "categorical".'
            )  
        return class_mode

    if class_mode=='none':
        if class_names is not None:
            raise ValueError(
                'You can only pass `class_names` if '
                '`class_mode` is "int" or "categorical".'
                f'Received: class_mode={labels}, and '
                f"class_names={class_names}"
            )                            
        return lambda x: x
    elif class_mode=='raw':
        if class_names is not None:
            raise ValueError(
                'You can only pass `class_names` if '
                '`class_mode` is "int" or "categorical".'
                f'Received: class_mode={labels}, and '
                f"class_names={class_names}"
            )                            
        return lambda x: tf.cast(x, "float32")
    elif class_mode=='int':
        if class_names is None:
            raise ValueError(
                'You have to `class_names` if '
                '`class_mode` is "int" or "categorical".'
                f'Received: class_mode={labels}, and '
                f"class_names={class_names}"
            )
        return lambda x: class_names.index(x)
    elif class_mode=='categorical':
        if class_names is None:
            raise ValueError(
                'You have to `class_names` if '
                '`class_mode` is "int" or "categorical".'
                f'Received: class_mode={labels}, and '
                f"class_names={class_names}"
            ) 
        num_classes = len(class_names)
        return lambda x: tf.one_hot(class_names.index(x), num_classes)                     
    else:
        raise ValueError(
            '`class_mode` argument must be callable or '
            'one of "int", "categorical", or "raw". '
            f'Received: class_mode={class_mode}'
        )                       
    

def dataset_from_dataframe(
    dataframe,
    colname_input,
    colname_target,
    load_fun_input,
    load_fun_target,
    batch_size=None,
    shuffle=False,
    dictname_input=None,
    dictname_target=None,
    seed=None,
):
    if seed is None:
        seed = np.random.randint(1e6)
    if dictname_input is None:
        dictname_input = colname_input
    if dictname_target is None:
        dictname_target = colname_target
    
    if isinstance(dataframe, str):
        dataframe = pandas.read_csv(dataframe)

    dataset = tf.data.Dataset.from_tensor_slices({
        dictname_input : dataframe[colname_input],
        dictname_target: dataframe[colname_target],
    })
    dataset = dataset.map( lambda x: {
            dictname_input : load_fun_input(x[dictname_input]),
            dictname_target: load_fun_target(x[dictname_target]),
    }, num_parallel_calls = tf.data.AUTOTUNE)
    
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    if batch_size is not None:
        if shuffle:
            # Shuffle locally at each iteration
            dataset = dataset.shuffle(buffer_size=batch_size * 8, seed=seed)
        dataset = dataset.batch(batch_size)
    elif shuffle:
        dataset = dataset.shuffle(buffer_size=1024, seed=seed)
    
    return dataset


def image_dataset_from_dataframe(
    dataframe,
    root_path='./',
    class_names=None,
    class_mode="categorical",
    color_mode="rgb",
    batch_size=32,
    shuffle=True,
    seed=None,
    interpolation="bilinear",
    image_size=None,
    colname_image='image',
    colname_class='class',
):
    """Generates a `tf.data.Dataset` from a dataframe.

    Then calling `image_dataset_from_dataframe(dataframe)`
    will return a `tf.data.Dataset` that yields batches of
    images from the subdirectories `class_a` and `class_b`, together with labels
    0 and 1 (0 corresponding to `class_a` and 1 corresponding to `class_b`).

    Args:
      dataframe: Dataframe with list of images.    
      class_mode: String describing the encoding of classes. Options are:
          - 'raw': means that the classes are not encoded
          - 'int': means that the classes are encoded as integers
              (e.g. for `sparse_categorical_crossentropy` loss).
          - 'categorical' means that the classes are
              encoded as a categorical vector
              (e.g. for `categorical_crossentropy` loss).
          - None (no labels).
      class_names: Only valid if "class_mode" is not "raw". This is the explicit
          list of class names. Used for the encoding
          (otherwise alphanumerical order is used).
      color_mode: One of "grayscale", "rgb", "rgba". Default: "rgb".
          Whether the images will be converted to
          have 1, 3, or 4 channels.
      batch_size: Size of the batches of data. Default: 32.
        If `None`, the data will not be batched
        (the dataset will yield individual samples).
      shuffle: Whether to shuffle the data. Default: True.
          If set to False, sorts the data in alphanumeric order.
      seed: Optional random seed for shuffling and transformations.
      image_size: Size to resize images to after they are read from disk,
              specified as `(height, width)`.
              When it is `None` no resizeing is computed. Defaults to `None`.       
      interpolation: String, the interpolation method used when resizing images.
              Defaults to `bilinear`. Supports `bilinear`, `nearest`, `bicubic`,
                `area`, `lanczos3`, `lanczos5`, `gaussian`, `mitchellcubic`.
    Returns:
      A `tf.data.Dataset` object.

    Rules regarding labels format:

      - if `class_mode` is `int`, the labels are an `int32` tensor of shape
        `(batch_size,)`.
      - if `class_mode` is `raw`, the classes are converted.
      - if `class_mode` is `categorical`, the labels are a `float32` tensor
        of shape `(batch_size, num_classes)`, representing a one-hot
        encoding of the class index.

    Rules regarding number of channels in the yielded images:

      - if `color_mode` is `grayscale`,
        there's 1 channel in the image tensors.
      - if `color_mode` is `rgb`,
        there are 3 channels in the image tensors.
      - if `color_mode` is `rgba`,
        there are 4 channels in the image tensors.
    """
   
    if isinstance(dataframe, str):
        dataframe = pandas.read_csv(dataframe)
    
    if class_mode in {"int", "categorical"}:
        if class_names is None:
            class_names = sorted(list(set(dataframe[colname_class])))
        dataframe = dataframe[dataframe[colname_class].isin(class_names)]
                                 
    load_fun_target = _get_fun_load_class(class_mode, class_names)
                                 
    if color_mode == "rgb":
        num_channels = 3
    elif color_mode == "rgba":
        num_channels = 4
    elif color_mode == "grayscale":
        num_channels = 1
    else:
        raise ValueError(
            '`color_mode` must be one of {"rgb", "rgba", "grayscale"}. '
            f"Received: color_mode={color_mode}"
        ) 
    interpolation = image_utils.get_interpolation(interpolation)
    
    load_fun_input = lambda x: load_img(os.path.join(root_path,x), 
                                        num_channels, image_size, interpolation)
                                 
    
    dataset = dataset_from_dataframe(
        dataframe=dataframe,
        colname_input=colname_image,
        colname_target=colname_class,
        load_fun_input=load_fun_input,
        load_fun_target=load_fun_target,
        batch_size=batch_size,
        shuffle=shuffle,
        dictname_input=IMAGES,
        dictname_target=LABELS,
        seed=seed,
    )
    
    # Users may need to reference `class_names`.
    dataset.class_names = class_names
    # Include file paths for images as attribute.
    dataset.root_path = root_path
    return dataset
