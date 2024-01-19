import os

os.environ["KERAS_BACKEND"] = "jax"
from pathlib import Path

import keras
from dataset import ICDARPyDataset
from losses import DBLoss
from model import DBHead
from model import FPNModel
from model import get_backbone_model

if __name__ == "__main__":
    base_data_dir = Path("text_localization")
    train_images_dir = base_data_dir / "icdar_c4_train_imgs"
    test_images_dir = base_data_dir / "ch4_test_images"
    train_labels_path = base_data_dir / "train_icdar2015_label.txt"
    text_labels_dir = base_data_dir / "test_icdar2015_label.txt"

    dataset = ICDARPyDataset(train_images_dir, train_labels_path)

    input_shape = (None, None, 3)
    inputs = keras.Input(shape=input_shape)
    x = get_backbone_model(input_shape)(inputs)
    x = FPNModel(out_channels=256)(x)
    outputs = DBHead(in_channels=256)(x, training=True)
    model = keras.Model(inputs=inputs, outputs=outputs)

    db_loss = DBLoss()

    # Forward propagate a single sample
    sample = dataset[0]
    outputs = model(sample["image"])

    # Compute the loss
    loss = db_loss(
        y_true=[sample["shrink_map"], sample["threshold_map"]],
        y_pred=outputs,
        mask=[sample["shrink_mask"], sample["threshold_mask"]],
    )

    # TODO: Backpropagate the loss
