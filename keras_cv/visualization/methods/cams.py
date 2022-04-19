import tensorflow as tf


@tf.function(experimental_relax_shapes=True, experimental_follow_type_hints=True)
def gradcam(
    model,
    inputs,
    indices=None,
    indices_batch_dims=-1,
    indices_axis=-1,
    positional_axis=(-3, -2),
):
    """Computes the Grad-CAM Visualization Method.

    This method expects `inputs` to be a batch of positional signals of shape
    `BHWC`, and will return a tensor of shape `BH'W'L`, where `(H', W')` are
    the sizes of the visual receptive field in the explained activation layer
    and `L` is the number of labels represented within the model's output
    logits.

    If `indices` is passed, the specific logits indexed by elements in this
    tensor are selected before the gradients are computed, effectivelly
    reducing the columns in the jacobian, and the size of the output
    explaining map.

    """
    with tf.GradientTape(watch_accessed_variables=False) as tape:
        tape.watch(inputs)
        activations, logits = model(inputs, training=False)

        if indices is not None:
            logits = tf.gather(
                logits, indices, batch_dims=indices_batch_dims, axis=indices_axis
            )

    dlda = tape.batch_jacobian(logits, activations)
    weights = tf.reduce_mean(dlda, axis=positional_axis)
    maps = tf.einsum("b...k,bck->b...c", activations, weights)

    return logits, maps
