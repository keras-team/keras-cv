import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow import keras

from keras_cv import bounding_box


def curry_map_function(bounding_box_format, img_size):
    """Mapping function to create batched image and bbox coordinates"""

    resizing = keras.layers.Resizing(
        height=img_size[0], width=img_size[1], crop_to_aspect_ratio=False
    )

    # TODO(lukewood): update `keras.layers.Resizing` to support bounding boxes.
    def apply(inputs):
        inputs["image"] = resizing(inputs["image"])
        inputs["objects"]["bbox"] = bounding_box.convert_format(
            inputs["objects"]["bbox"],
            images=inputs["image"],
            source="rel_yxyx",
            target=bounding_box_format,
        )

        bounding_boxes = inputs["objects"]["bbox"]
        labels = tf.cast(inputs["objects"]["label"], tf.float32)
        labels = tf.expand_dims(labels, axis=-1)
        bounding_boxes = tf.concat([bounding_boxes, labels], axis=-1)
        return {"images": inputs["image"], "bounding_boxes": bounding_boxes}

    return apply


def load(
    split, bounding_box_format, batch_size=None, shuffle=True, img_size=(512, 512)
):
    """Loads the PascalVOC 2007 dataset.

    Usage:
    ```python
    dataset, ds_info = keras_cv.loaders.pascal_voc.load(
        split="train", bounding_box_format="xywh", batch_size=9
    )
    ```

    Args:
        split:
        bounding_box_format:
        batch_size:
        shuffle:
        img_size:

    Returns:
        tf.data.Dataset containing PascalVOC.  Each entry is a dictionary containing
        keys {"images": images, "bounding_boxes": bounding_boxes} where images is a
        Tensor of shape [batch, H, W, 3] and bounding_boxes is a `tf.RaggedTensor` of
        shape [batch, None, 5].
    """
    dataset, dataset_info = tfds.load(
        "voc/2007", split=split, shuffle_files=shuffle, with_info=True
    )
    dataset = dataset.map(
        curry_map_function(bounding_box_format=bounding_box_format, img_size=img_size),
        num_parallel_calls=tf.data.AUTOTUNE,
    )
    dataset = dataset.shuffle(8 * batch_size)

    if batch_size is not None:
        dataset = dataset.apply(
            tf.data.experimental.dense_to_ragged_batch(batch_size=batch_size)
        )
    return dataset, dataset_info
