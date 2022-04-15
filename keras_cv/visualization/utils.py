import tensorflow as tf


def get_nested_layer(model, name):
    """Get nested layer in model.

    Example:
    ```py
    model = tf.keras.Sequential([
        tf.keras.applications.ResNet101V2(include_top=False, pooling='avg'),
        tf.keras.layers.Dense(10, activation='softmax', name='predictions')
    ])

    pooling_layer = get_nested_layer(model, 'resnet101v2.avg_pool')
    ```
    """
    for n in name.split("."):
        model = model.get_layer(n)

    return model


def collect_endpoints(model, endpoints):
    """Collect (intermediate) endpoints in a model."""
    endpoints_ = []

    for ep in endpoints:
        if isinstance(ep, str):
            ep = {"name": ep}

        layer = ep["name"]
        link = ep.get("link", "output")
        bind = ep.get("bind", 0)

        endpoints_.append(
            get_nested_layer(model, layer).get_input_at(bind)
            if link == "input"
            else get_nested_layer(model, layer).get_output_at(bind)
        )

    return endpoints_


def normalize(x, axis=(-3, -2)):
    """Normalize a positional signal between 0 and 1."""
    x = tf.convert_to_tensor(x)
    x -= tf.reduce_min(x, axis=axis, keepdims=True)

    return tf.math.divide_no_nan(x, tf.reduce_max(x, axis=axis, keepdims=True))
