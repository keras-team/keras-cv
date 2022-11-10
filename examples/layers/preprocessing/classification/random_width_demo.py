"""
random_width_demo.py shows how to use the RandomWidth preprocessing layer for
object detection.
"""
import demo_utils
import tensorflow as tf

from keras_cv.layers import preprocessing


def main():
    ds = demo_utils.load_oxford_dataset()
    random_width = preprocessing.RandomWidth(
        factor=(0.0, 1.0), interpolation="bilinear"
    )
    ds = ds.map(random_width, num_parallel_calls=tf.data.AUTOTUNE)
    demo_utils.visualize_dataset(ds)


if __name__ == "__main__":
    main()
