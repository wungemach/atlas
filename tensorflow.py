import tensorflow as tf
import numpy as np

input = tf.constant([1, 2, 3, 4, 5, 6, 7])

green_box = tf.map_fn(lambda img: tf.image.central_crop(img, 0.5), input, parallel_iterations=8, name="crop1", axis=0)

print(green_box.shape)
