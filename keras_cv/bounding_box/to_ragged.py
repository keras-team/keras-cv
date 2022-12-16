from keras_cv.bounding_box import validate

def to_ragged(bounding_boxes):
    """converts bounding boxes to Ragged tensors.

    Args:
        bounding_boxes: bounding boxes in KerasCV dictionary format.
    """
    info = validate(bounding_boxes)

    if info['ragged']:
        return bounding_boxes

    bounding_boxes['boxes'] = tf.RaggedTensor.from_tensor(bounding_boxes['boxes'])
    bounding_boxes['classes'] = tf.RaggedTensor.from_tensor(bounding_boxes['classes'])
    return bounding_boxes
