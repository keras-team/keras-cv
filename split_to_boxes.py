def split_to_boxes(boxes):
    b = boxes[..., :4]
    classes = boxes[..., 4]
    result = {"boxes": b.tolist(), "classes": classes.tolist()}

    if boxes.shape[-1] == 6:
        result["confidence"] = boxes[..., 5].tolist()

    print(result)
