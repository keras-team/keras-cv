# TODO Uncomment when ready from keras_cv.callbacks import WaymoDetectionMetrics
import tensorflow as tf

from keras_cv import layers
from keras_cv.datasets.waymo import load
from keras_cv.datasets.waymo import build_tensors_from_wod_frame
from keras_cv.datasets.waymo import build_tensors_for_augmentation

TRAINING_RECORD_PATH = "./wod-records"#"gs://waymo_open_dataset_v_1_0_0_individual_files/training"
EVALUATION_RECORD_PATH = "./wod-records"#"gs://waymo_open_dataset_v_1_0_0_individual_files/validation"
TENSORBOARD_LOGS_PATH = "./logs"

### Load the training dataset
train_ds = load(TRAINING_RECORD_PATH)

train_ds = train_ds.map(build_tensors_for_augmentation)

### Augment the training data
AUGMENTATION_LAYERS = [
    # layers.BaseAugmentationLayer3D(),
    # TODO need to add real augmentation layers here.
]

for layer in AUGMENTATION_LAYERS:
    train_ds = layer(train_ds)


### Load the evaluation dataset
eval_ds = load(EVALUATION_RECORD_PATH, simple_transformer, output_signature)


### Load and compile the model
strategy = tf.distribute.MirroredStrategy()
with strategy.scope():
    model = None  # TODO Need to import model and instantiate it here

model.compile(optimizer="adam", loss=None)  # TODO need to specify appropriate loss here


### Fit the model with a callback to log scores on our evaluation dataset
model.fit(
    train_ds,
    callbacks=[
        WaymoDetectionMetrics(eval_ds),
        keras.callbacks.TensorBoard(TENSORBOARD_LOGS_PATH),
    ],
)
