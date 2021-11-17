"""Shared constants for use with bounding boxes."""
# These are the dimensions used in Tensors to represent each corresponding side.
LEFT, TOP, RIGHT, BOTTOM = 0, 1, 2, 3
# Class is held in the 4th index
CLASS = 4
# Confidence exists only on y_pred, and is in the 5th index.
CONFIDENCE = 5
