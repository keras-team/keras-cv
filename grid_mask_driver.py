"""gridmask_demo.py shows how to use the GridMask preprocessing layer.

Operates on the oxford_flowers102 dataset.  In this script the flowers
are loaded, then are passed through the preprocessing layers.
Finally, they are shown using matplotlib.
"""

import matplotlib.pyplot as plt
import tensorflow as tf

from keras_cv.utils import fill_utils

IMG_SHAPE = (2, 224, 224)

img = tf.ones(IMG_SHAPE)

# %%
ratio = 0.6

img_h, img_w = 224, 224
img_w = tf.cast(img_w, tf.float32)
img_h = tf.cast(img_h, tf.float32)

squared_w = tf.square(img_w)
squared_h = tf.square(img_h)
mask_hw = tf.math.ceil(tf.sqrt(squared_w + squared_h))
mask_hw = tf.cast(mask_hw, tf.int32)

d = tf.random.uniform(
    shape=[],
    minval=tf.math.minimum(img_h * 0.5, img_w * 0.3),
    maxval=tf.math.maximum(img_h * 0.5, img_w * 0.3) + 1,
)
space = ratio * d

d = tf.cast(d, tf.int32)
space = tf.cast(space, tf.int32)
square_l = d - space

delta_x = tf.random.uniform([], minval=0, maxval=space, dtype=tf.int32)
delta_y = tf.random.uniform([], minval=0, maxval=space, dtype=tf.int32)

gridsize = mask_hw // d + 1
gridrange = tf.range(1, gridsize)
d_range = gridrange * d
x1 = d_range - delta_x
x0 = x1 - square_l
y1 = d_range - delta_y
y0 = y1 - square_l

x0, y0 = tf.meshgrid(x0, y0)
x1, y1 = tf.meshgrid(x1, y1)
corners0 = tf.stack([x0, y0], axis=-1)
corners1 = tf.stack([x1, y1], axis=-1)
corners0 = tf.reshape(corners0, [-1, 2])
corners1 = tf.reshape(corners1, [-1, 2])
corners = tf.concat([corners0, corners1], axis=1)

mask_shape = (tf.shape(corners)[0], mask_hw, mask_hw)
masks = fill_utils.rectangle_masks(mask_shape, corners)
mask = tf.reduce_any(masks, axis=0)

# TODO: Rotate mask
# TODO: Center crop mask

plt.imshow(mask)
plt.show()

#%%
ratio = 0.6

batch_size, img_h, img_w = tf.shape(img)
img_w = tf.cast(img_w, tf.float32)
img_h = tf.cast(img_h, tf.float32)

squared_w = tf.square(img_w)
squared_h = tf.square(img_h)
mask_hw = tf.math.ceil(tf.sqrt(squared_w + squared_h))
mask_hw = tf.cast(mask_hw, tf.int32)
mask = tf.zeros((batch_size, mask_hw, mask_hw), dtype=tf.bool)

d = tf.random.uniform(
    shape=[batch_size],
    minval=tf.math.minimum(img_h * 0.5, img_w * 0.3),
    maxval=tf.math.maximum(img_h * 0.5, img_w * 0.3) + 1,
)
space = ratio * d

start_xy = tf.random.uniform([batch_size, 2], minval=0, maxval=1, dtype=tf.float32)
start_xy = start_xy * tf.expand_dims(d, 1)

start_xy = tf.cast(start_xy, tf.int32)
d = tf.cast(d, tf.int32)
space = tf.cast(space, tf.int32)

start_xy
d
space

#%%
"""
mask2.png
Y
start 51
Num blocks 4
51 101
135 185
219 269
303 353
X
start 72
Num blocks 4
72 122
156 206
240 290
324 374

mask3.png
Y
start 7
Num blocks 5
7 55
87 135
167 215
247 295
327 375
X
start 41
Num blocks 5
41 89
121 169
201 249
281 329
361 403
"""
ratio = 0.6

batch_size, img_h, img_w = tf.shape(img)
img_w = tf.cast(img_w, tf.float32)
img_h = tf.cast(img_h, tf.float32)

mask_hw = 403
mask = tf.zeros((batch_size, mask_hw, mask_hw), dtype=tf.bool)

d = tf.constant([84, 80], tf.float32)
space = ratio * d
space = tf.cast(space, tf.int32)

start_xy = tf.constant(
    [
        [72, 51],  # mask2
        [41, 7],  # mask3
    ]
)

start_xy = tf.cast(start_xy, tf.int32)
d = tf.cast(d, tf.int32)

start_xy
d
space

start_xy


###
gridsize_keep = mask_hw // d
gridsize_mask = gridsize_keep + 1
gridsize_mask

#%%
ratio = 0.6

batch_size, img_h, img_w = tf.shape(img)
img_w = tf.cast(img_w, tf.float32)
img_h = tf.cast(img_h, tf.float32)

squared_w = tf.square(img_w)
squared_h = tf.square(img_h)
mask_hw = tf.math.ceil(tf.sqrt(squared_w + squared_h))
mask_hw = tf.cast(mask_hw, tf.int32)
mask = tf.zeros((batch_size, mask_hw, mask_hw), dtype=tf.bool)

d = tf.random.uniform(
    shape=[batch_size],
    minval=tf.math.minimum(img_h * 0.5, img_w * 0.3),
    maxval=tf.math.maximum(img_h * 0.5, img_w * 0.3) + 1,
)
space = ratio * d
space = tf.cast(space, tf.int32)

start_xy = tf.random.uniform([batch_size, 2], minval=0, maxval=1, dtype=tf.float32)
start_xy = start_xy * tf.expand_dims(d, 1)

start_xy = tf.cast(start_xy, tf.int32)
d = tf.cast(d, tf.int32)

gridsize_keep = mask_hw // d
gridsize_mask = gridsize_keep + 1

d
start_xy

#%%
# gridmask = preprocessing.GridMask(
#     ratio=0.6, gridmask_size_ratio=0.8, rate=0.8
# )
# z = gridmask._grid_mask(img, training=True)
# plt.imshow(z)
# plt.show()
