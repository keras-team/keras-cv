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
from functools import partial
import keras
from keras.utils import dataset_utils
from keras.utils import image_utils

from keras_cv.layers.preprocessing.base_image_augmentation_layer import IMAGES
from keras_cv.layers.preprocessing.base_image_augmentation_layer import LABELS
from keras_cv.layers.preprocessing.base_image_augmentation_layer import BOUNDING_BOXES 
from keras_cv.layers.preprocessing.base_image_augmentation_layer import KEYPOINTS 
from keras_cv.layers.preprocessing.base_image_augmentation_layer import SEGMENTATION_MASKS
from keras_cv.layers.preprocessing import Rescaling

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

     
def _load_img(path, root_path='./', num_channels=3):
    """Load an image from a path and resize it."""
    #TODO path = tf.io.gfile.join(root_path, path)
    img = tf.io.read_file(path)
    img = tf.image.decode_image(
        img, channels=num_channels, expand_animations=False
    )
    return img

def _dict_to_tuple_fun(dat, dictname_input, dictname_target, max_boxes=None):
    x = dat[dictname_input]
    y = dat[dictname_target]
    if dictname_target==BOUNDING_BOXES:
        y = bounding_box.to_dense(y, max_boxes=max_boxes)
    return x,y



def _get_fun_load_class(x, class_mode, num_classes):
    if class_mode=='none':                               
        return x
    elif class_mode=='raw':                        
        return tf.cast(x, "float32")
    elif class_mode=='int':
        return tf.cast(x, "int64")
    elif class_mode=='categorical':
        return tf.one_hot(x, num_classes)
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
    shuffle=False,
    seed=None,
    pre_batching_processing=None,
    batch_size=None,
    post_batching_processing=None,
    dictname_input=IMAGES,
    dictname_target=None,
    dict_to_tuple=True,
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
    
    if shuffle:
        dataset = dataset.shuffle(buffer_size=len(dataframe), seed=seed, reshuffle_each_iteration=True)
        
    #for a in dataset.take(9):
    #    for k in a:
    #        print(k, a[k].shape, a[k].dtype, a[k])
    
    dataset = dataset.map( lambda x: {
            dictname_input : load_fun_input(x[dictname_input]),
            dictname_target: load_fun_target(x[dictname_target]),
    }, num_parallel_calls = tf.data.AUTOTUNE)
    
    if pre_batching_processing is not None:
        dataset = dataset.map(pre_batching_processing, num_parallel_calls = tf.data.AUTOTUNE)
            
    if batch_size is not None:
        try:
            dataset = dataset.ragged_batch(batch_size)
            #dataset = dataset.batch(batch_size)
        except:
            dataset = dataset.apply(tf.data.experimental.dense_to_ragged_batch(batch_size))
    
    if post_batching_processing is not None:
        dataset = dataset.map(post_batching_processing, num_parallel_calls = tf.data.AUTOTUNE)   
        
    if dict_to_tuple:
        dataset = dataset.map(lambda x: _dict_to_tuple_fun(x,dictname_input, dictname_target),
                              num_parallel_calls = tf.data.AUTOTUNE)
    
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset


def image_classification_dataset_from_dataframe(
    dataframe,
    root_path='./',
    class_names=None,
    class_mode="categorical",
    color_mode="rgb",
    batch_size=None,
    shuffle=True,
    seed=None,
    pre_batching_processing=None,
    post_batching_processing=None,
    include_rescaling=True,
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
      class_names: Only valid if "class_mode" is 'int' or 'categorical'. This is the explicit
          list of class names esed for the encoding.
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
      pre_batching_processing: ???      
      post_batching_processing: ???.
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
    
    num_classes = 0
    if class_mode in {"int", "categorical"}:
        if class_names is None:
            class_names = sorted(list(set(dataframe[colname_class])))
        
        dataframe = dataframe.copy()
        dataframe = dataframe[dataframe[colname_class].isin(class_names)]
        dataframe[colname_class] = [class_names.index(_) for _ in dataframe[colname_class]]
        num_classes = len(class_names)
        for index_class, name_class in enumerate(class_names):
            num_img = np.sum(dataframe[colname_class]==index_class)
            print(f"For class '{name_class}', there are {num_img} images.", flush=True)
        
    elif class_names:
        raise ValueError(
            'You can only pass `class_names` if '
            '`class_mode` is "int" or "categorical".'
            f'Received: class_mode={class_mode}, and '
            f"class_names={class_names}"
        )
    
    
    if (not isinstance(class_mode, str)) and callable(class_mode):
        load_fun_target = class_mode
    else:
        load_fun_target = partial(_get_fun_load_class, class_mode=class_mode, num_classes=num_classes)
    
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
    
    load_fun_input = partial(_load_img, root_path=root_path, num_channels=num_channels)
    
    
    if include_rescaling:
        if post_batching_processing is None:
            post_batching_processing = Rescaling(1 / 255.0)
        else:
            post_batching_processing = keras.Sequential(
                layers=[post_batching_processing, Rescaling(1 / 255.0)])
            
    dataset = dataset_from_dataframe(
        dataframe=dataframe,
        colname_input=colname_image,
        colname_target=colname_class,
        load_fun_input=load_fun_input,
        load_fun_target=load_fun_target,
        shuffle=shuffle,
        seed=seed,
        pre_batching_processing=pre_batching_processing,
        batch_size=batch_size,
        post_batching_processing=post_batching_processing,
        dictname_input=IMAGES,
        dictname_target=LABELS,
        dict_to_tuple=True,
    )
    
    # Users may need to reference `class_names`, `batch_size` and `root_path`
    dataset.class_names = class_names
    dataset.batch_size = batch_size
    dataset.root_path = root_path
    return dataset
