def parse_imagenet_example(example, image_size):
    """Function to parse a TFRecord example into an image and label"""
    # Read example
    image_key = "image/encoded"
    label_key = "image/class/label"
    keys_to_features = {
        image_key: tf.io.FixedLenFeature((), tf.string, ""),
        label_key: tf.io.FixedLenFeature([], tf.int64, -1),
    }
    parsed = tf.io.parse_single_example(example, keys_to_features)

    # Decode and resize image
    image_bytes = tf.reshape(parsed[image_key], shape=[])
    image = tf.io.decode_jpeg(image_bytes, channels=3)
    image = layers.Resizing(
        width=image_size[0], height=image_size[1], crop_to_aspect_ratio=True
    )(image)

    # Decode label
    label = tf.cast(tf.reshape(parsed[label_key], shape=()), dtype=tf.int32) - 1
    label = tf.one_hot(label, 1000)

    return {"images": image, "labels": label}


def load(
    split,
    tfrecords_path,
    batch_size=None,
    shuffle=True,
    shuffle_buffer=None,
    reshuffle_each_iteration=False,
    img_size=(512, 512),
):
    """Loads the ImageNet dataset from TFRecords

    Usage:
    ```python
    dataset, ds_info = keras_cv.datasets.imagenet.load(
        split="train", tfrecords_path="gs://my-bucket/imagenet-tfrecords"
    )
    ```

    Args:
        split: the split to load.  Should be one of "train" or "validation."
        tfrecords_path: the path to your preprocessed ImageNet TFRecords.
            See keras_cv/datasets/imagenet/README.md for preprocessing instructions.
        batch_size: how many instances to include in batches after loading
        shuffle: whether or not to shuffle the dataset.  Defaults to True.
        shuffle_buffer: the size of the buffer to use in shuffling.
        reshuffle_each_iteration: whether to reshuffle the dataset on every epoch.
            Defaults to False.
        img_size: the size to resize the images to. Defaults to (512, 512).

    Returns:
        tf.data.Dataset containing ImageNet.  Each entry is a dictionary containing
        keys {"images": images, "labels": label} where images is a
        Tensor of shape [batch, H, W, 3] and bounding_boxes is a Tensor of shape
        [batch, 1000].
    """

    num_splits = 1024 if split == "train" else 128
    filenames = [
        f"{tfrecords_path}/{split}-{i:05d}-of-{num_splits:05d}"
        for i in range(0, num_splits)
    ]

    dataset = tf.data.TFRecordDataset(filenames=filenames)

    dataset = train_dataset.map(
        lambda example: parse_imagenet_example(example, img_size),
        num_parallel_calls=tf.data.AUTOTUNE,
    )

    if shuffle:
        if not batch_size and not shuffle_buffer:
            raise ValueError(
                "If `shuffle=True`, either a `batch_size` or `shuffle_buffer` must be "
                "provided to `keras_cv.datasets.imagenet.load().`"
            )
        shuffle_buffer = shuffle_buffer or 8 * batch_size
        dataset = dataset.shuffle(
            shuffle_buffer, reshuffle_each_iteration=reshuffle_each_iteration
        )

    if batch_size is not None:
        dataset = dataset.batch(FLAGS.batch_size)

    return dataset
