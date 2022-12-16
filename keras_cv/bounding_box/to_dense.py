from keras_cv.bounding_box import validate

def to_dense(bounding_boxes):
    """to_dense converts bounding boxes to Dense tensors

    Args:
        bounding_boxes: bounding boxes in KerasCV dictionary format.
    """
    info = validate(bounding_boxes)

    # Already running in masked mode
    if not info['ragged']:
        return bounding_boxes

    bounding_boxes['classes'] = bounding_boxes['classes'].to_tensor(-1)
    bounding_boxes['boxes'] = bounding_boxes['boxes'].to_tensor(-1)
    return bounding_boxes
