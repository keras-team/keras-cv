import time

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from keras_cv.utils import fill_utils


@tf.function
def map_fill_rectangle(images, center_x, center_y, height, width, fill):
    images = tf.map_fn(
        lambda x: fill_utils.fill_rectangle(*x),
        (
            images,
            center_x,
            center_y,
            height,
            width,
            fill,
        ),
        fn_output_signature=tf.TensorSpec.from_tensor(images[0]),
    )
    return images


batch_fill_rectangle = tf.function(fill_utils.batch_fill_rectangle)


def time_batch_fill(images, cx, cy, height, width, fill, n):
    times = []
    for _ in range(n):
        st = time.time()
        images = batch_fill_rectangle(images, cx, cy, height, width, fill)
        total = time.time() - st
        times.append(total)
    return times, images


def time_map_fill(images, cx, cy, height, width, fill, n):
    times = []
    for _ in range(n):
        st = time.time()
        images = map_fill_rectangle(images, cx, cy, height, width, fill)
        total = time.time() - st
        times.append(total)
    return times, images


#%%
n = 1
batch_sizes = [2 ** i for i in range(13)]
h, w = 32, 32
rh, rw = h // 2, w // 2

batch_times = []
map_times = []
for batch_size in batch_sizes:
    batch_shape = (batch_size, h, w, 1)
    images = tf.ones(batch_shape)

    cx = tf.fill([batch_size], rw)
    cy = tf.fill([batch_size], rh)
    height = tf.fill([batch_size], rh)
    width = tf.fill([batch_size], rw)
    fill = tf.zeros_like(images)

    time_batch_fill(images, cx, cy, height, width, fill, 1)
    time_map_fill(images, cx, cy, height, width, fill, 1)

    batch_times_i, bi = time_batch_fill(images, cx, cy, height, width, fill, n=n)
    single_times_i, si = time_map_fill(images, cx, cy, height, width, fill, n=n)
    batch_times.append(batch_times_i)
    map_times.append(single_times_i)

plt.imshow(bi[0])
plt.show()
plt.imshow(si[0])
plt.show()

# batch_sizes x n
batch_times = np.array(batch_times) * 1000
map_times = np.array(map_times) * 1000

plt.plot(batch_sizes, batch_times.mean(1), label="batch fill")
plt.plot(batch_sizes, map_times.mean(1), label="map fill")
plt.title(
    "Fill batch of {}x{} rectangles into batch of {}x{} images.".format(rh, rw, h, w)
)
plt.xlabel("Batch size")
plt.ylabel("Fill time [milliseconds]")
plt.legend()
plt.savefig("fill_benchmark.png")
plt.show()

print(batch_times[:, 0])
print(map_times[:, 0])
