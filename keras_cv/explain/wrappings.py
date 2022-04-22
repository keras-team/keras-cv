import tensorflow as tf

from keras_cv.explain import methods
from keras_cv.explain.utils import collect_endpoints


def explain(
    method,
    model,
    inputs,
    indices=None,
    indices_batch_dims=-1,
    indices_axis=-1,
    positional_axis=(-3, -2),
    pooling_layer=None,
    logits_layer=None,
    postprocessing=tf.nn.relu,
    normalized_maps=True,
):
    if logits_layer is None:
        logits_layer = model.layers[-1]
    if pooling_layer is None:
        pooling_layer = model.layers[-2].name
    if isinstance(pooling_layer, str):
        pooling_layer = {"name": pooling_layer, "link": "input"}

    endpoints = [pooling_layer, logits_layer]
    model_eps = tf.keras.Model(
        inputs=model.inputs,
        outputs=collect_endpoints(model, endpoints),
    )

    inputs = tf.convert_to_tensor(inputs)
    indices = tf.convert_to_tensor(indices)

    logits, maps = method(
        model_eps,
        inputs,
        indices=indices,
        indices_batch_dims=indices_batch_dims,
        indices_axis=indices_axis,
        positional_axis=positional_axis,
    )

    if postprocessing is not None:
      maps = postprocessing(maps)

    if normalized_maps:
        maps = _normalize(maps, axis=positional_axis)

    return logits, maps


def gradcam(
    model,
    inputs,
    indices=None,
    indices_batch_dims=-1,
    indices_axis=-1,
    positional_axis=(-3, -2),
    pooling_layer=None,
    logits_layer=None,
    postprocessing=tf.nn.relu,
    normalized_maps=True,
):
    """Apply Grad-CAM to produce activation maps of units with respect to an
    intermediate positional signal.

    Args:
        model: The `tf.keras.Model` being evaluated.
        inputs: The input data for the model.
        indices: The indices of the units in `logits_layer` being explained.
          If none, an activation map is computed for each unit.
        indices_batch_dims: The dimensions set as `batch` when gathering units
          described by `indices`. Ignored if `indices` is None.
        indices_axis: The axis from which to gather units described by `indices`.
          Ignored if `indices` is None.
        positional_axis: The axes containing the positional visual info. We
          assume `inputs` to contain 2D images or videos in the shape
          `(B1, B2, ..., BN, H, W, 3)`. For 3D image data, set
          `positional_axis` to `(1, 2, 3)` or `(-4, -3, -2)`.
        pooling_layer: Name of the pooling layer in the model. The jacobian
          of the output explaining units will be computed with respect to
          the input signal of this layer. This argument can also be an
          integer, a dictionary representing the intermediate signal or
          the pooling layer itself. If None is passed, the penultimate layer
          is assumed to be a GAP layer.
        logits_layer: Name of the logits layer in the model. The jacobian
          will be computed for the activation signal of units in this layer.
          This argument can also be an integer, a dictionary representing the
          output signal and the logits layer itself. If None is passed,
          the last layer is assumed to be the logits layer.
        postprocessing: A function to process the activation maps before
          normalization (most commonly adopted being `maximum(x, 0)` and
          `abs`).
        normalized_maps: A flag describing if the activation maps should be
          normalized into the [0, 1] interval.

    Returns:
        Logits: the output signal of the model, collected from `logits_layer`.
        Maps: the activation maps produced by Grad-CAM that explain the logits
        with respect to the intermediate positional signal in the model.
    """
    return explain(
        method=methods.gradcam,
        model=model,
        inputs=inputs,
        indices=indices,
        indices_batch_dims=indices_batch_dims,
        indices_axis=indices_axis,
        positional_axis=positional_axis,
        pooling_layer=pooling_layer,
        logits_layer=logits_layer,
        postprocessing=postprocessing,
        normalized_maps=normalized_maps,
    )


def _normalize(x, axis=(-3, -2)):
    """Normalize a positional signal between 0 and 1."""
    x = tf.convert_to_tensor(x)
    x -= tf.reduce_min(x, axis=axis, keepdims=True)

    return tf.math.divide_no_nan(x, tf.reduce_max(x, axis=axis, keepdims=True))
