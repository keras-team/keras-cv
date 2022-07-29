from keras_cv import models
import keras
from keras import layers
import tensorflow as tf
import tensorflow_datasets as tfds

NUM_CLASSES=2
BATCH_SIZE=32
EPOCHS=20

def load_cats_vs_dogs():
        train_ds, test_ds = tfds.load(
            "cats_vs_dogs", split=["train[:90%]", "train[90%:]"], as_supervised=True
        )
        resizing = layers.Resizing(150, 150)
        train = train_ds.map(
            lambda x, y: (resizing(x), tf.one_hot(y, 2)),
            num_parallel_calls=tf.data.AUTOTUNE,
        ).batch(BATCH_SIZE)
        test = test_ds.map(
            lambda x, y: (resizing(x), tf.one_hot(y, 2)),
            num_parallel_calls=tf.data.AUTOTUNE,
        ).batch(BATCH_SIZE)
        return train, test

densenet = models.DenseNet121(include_rescaling=True, include_top=False, weights='./imagenet.h5', pooling='avg')
densenet.trainable = False
model = keras.models.Sequential([densenet, layers.Dense(NUM_CLASSES, activation='softmax')])

train, test = load_cats_vs_dogs()
train, test = train.prefetch(tf.data.AUTOTUNE), test.prefetch(tf.data.AUTOTUNE)

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

model.fit(train, validation_data=test, batch_size=BATCH_SIZE, epochs=EPOCHS)
