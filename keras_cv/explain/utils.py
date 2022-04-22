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
        if isinstance(ep, int):
            endpoint = model.layers[ep].get_output_at(0)
        elif isinstance(ep, tf.keras.layers.Layer):
            endpoint = ep.get_output_at(0)
        else:
            if isinstance(ep, str):
                ep = {"name": ep}

            if not isinstance(ep, dict):
                raise ValueError(
                    f"Illegal type {type(ep)} for endpoint {ep}. Expected a "
                    "layer index (int), layer name (str) or a dictionary with "
                    "`name`, `link` and `bind` keys."
                )

            layer = ep["name"]
            link = ep.get("link", "output")
            bind = ep.get("bind", 0)

            endpoint = (
                get_nested_layer(model, layer).get_input_at(bind)
                if link == "input"
                else get_nested_layer(model, layer).get_output_at(bind)
            )

        endpoints_.append(endpoint)

    return endpoints_
