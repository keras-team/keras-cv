import tensorflow as tf


def validate(bounding_boxes):
    """validates that a given set of bounding boxes complies to KerasCV format.

    For a set of bounding boxes to be valid it must satisfy the following conditions:
    - `bounding_boxes` must be a dictionary
    - contains keys `"boxes"` and `"classes"`
    - each entry must have matching first two dimensions; representing the batch axis
        and the number of boxes per image axis.
    - either both `"boxes"` and `"classes"` are batched, or both are unbatched

    Additionally, one of the following must be satisfied:
    - `"boxes"` and `"classes"` are both Ragged
    - `"boxes"` and `"classes"` are both Dense
    - `"boxes"` and `"classes"` are unbatched

    Args:
        bounding_boxes: dictionary of bounding boxes according to KerasCV format.

    Raises:
        ValueError if any of the above conditions are not met
    """
    if not isinstance(bounding_boxes, dict):
        raise ValueError(
            "Expected `bounding_boxes` to be a dictionary, got "
            f"`bounding_boxes={bounding_boxes}`."
        )
    if not all([x in bounding_boxes for x in ["boxes", "classes"]]):
        raise ValueError(
            "Expected `bounding_boxes` to be a dictionary containing keys "
            "`'classes'` and `'boxes'`.  Got "
            f"`bounding_boxes.keys()={bounding_boxes.keys()}`."
        )

    boxes = bounding_boxes.get("boxes")
    classes = bounding_boxes.get("classes")
    info = {}

    is_batched = len(boxes.shape) == 3
    info['is_batched'] = is_batched
    info['ragged'] = isinstance(boxes, tf.RaggedTensor)

    if isinstance(boxes, tf.RaggedTensor) != isinstance(classes, tf.RaggedTensor):
        raise ValueError(
            "Either both `boxes` and `classes` "
            "should be Ragged, or neither should be ragged."
            f" Got `type(boxes)={type(boxes)}`, type(classes)={type(classes)}."
        )

    if not is_batched:
        if boxes.shape[:1] != classes.shape[:1]:
            raise ValueError(
                "Expected `boxes` and `classes` to have matching dimensions "
                "on the first axis when operating in unbatched mode. "
                f"Got `boxes.shape={boxes.shape}`, `classes.shape={classes.shape}`."
            )
        # No Ragged checks needed in unbatched mode.
        return info

    # Batched mode checks
    if boxes.shape[:2] != classes.shape[:2]:
        raise ValueError(
            "Expected `boxes` and `classes` to have matching dimensions "
            "on the first two axes when operating in batched mode. "
            f"Got `boxes.shape={boxes.shape}`, `classes.shape={classes.shape}`."
        )

    return info
