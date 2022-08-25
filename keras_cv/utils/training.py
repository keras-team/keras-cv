import tensorflow as tf


def convert_inputs_to_tf_dataset(x=None, y=None, sample_weight=None, batch_size=None):
    if sample_weight is not None:
        raise ValueError("Contrastive trainers do not yet support `sample_weight`.")

    if isinstance(x, tf.data.Dataset):
        if y is not None or batch_size is not None:
            raise ValueError(
                "When `x` is a `tf.data.Dataset`, please do not provide a value for "
                f"`y` or `batch_size`.  Got `y={y}`, `batch_size={batch_size}`."
            )
        return x

    # batch_size defaults to 32, as it does in fit().
    batch_size = batch_size or 32
    # Parse inputs
    inputs = x
    if y is not None:
        inputs = (x, y)

    # Construct tf.data.Dataset
    dataset = tf.data.Dataset.from_tensor_slices(inputs)
    if batch_size is not None:
        dataset = dataset.batch(batch_size)
    return dataset
