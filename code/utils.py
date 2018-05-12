import numpy as np
import tensorflow as tf

def dice_coefficient(predicted_mask, target_mask):
  tp = np.sum(np.logical_and(predicted_mask, target_mask))
  fp_fn = np.sum(np.logical_xor(predicted_mask, target_mask))
  if (2 * tp + fp_fn) == 0: return -1  # mask is entirely 0
  dice = 2 * tp / (2 * tp + fp_fn)
  return dice

def get_block_sizes(resnet_size):
  """
  Retrieves the number of block layers to use in the ResNet model, throwing an
  error if a non-standard size has been selected.

  Adapted from tensorflow/models/official/resnet/imagenet_main.py.

  Inputs:
  - resnet_size: A Python int representing the number of convolutional layers
    needed in the model.

  Outputs:
  - choice: A list of Python ints representing the block sizes to use in
    building the model.

  Raises:
  - ValueError: if invalid resnet_size is received.
  """
  choices = {
      18: [2, 2, 2, 2],
      34: [3, 4, 6, 3],
      50: [3, 4, 6, 3],
      101: [3, 4, 23, 3],
      152: [3, 8, 36, 3],
      200: [3, 24, 36, 3]
  }

  try:
    choice = choices[resnet_size]
  except KeyError:
    err = (f"Could not find layers for selected Resnet size.\n"
           f"Size received: {resnet_size}; sizes allowed: {choices.keys()}.")
    raise ValueError(err)
  else:
    return choice

def write_summary(value, tag, summary_writer, global_step):
  """Writes a single summary value to TensorBoard."""
  summary = tf.Summary()
  summary.value.add(tag=tag, simple_value=value)
  summary_writer.add_summary(summary, global_step)

def add_summary_image_triplet(inputs_op,
                              target_masks_op,
                              predicted_masks_op,
                              num_images=4):
  """
  Adds triplets of (input, target_mask, predicted_mask) images.

  Inputs:
  - inputs_op: A placeholder Tensor (dtype=tf.float32) with shape batch size
    by image dims e.g. (100, 233, 197) that represents the batch of inputs.
  - target_masks_op: A placeholder Tensor (dtype=tf.flota32) with shape batch
    size by mask dims e.g. (100, 233, 197) that represents the batch of target
    masks.
  - predicted_masks_op: A Tensor (dtype=tf.uint8) of the same shape as
    self.logits_op e.g. (100, 233, 197) of 0s and 1s.

  Outputs:
  - triplet: A Tensor of concatenated images with shape batch size by
    image height dim by 3 * image width dim e.g. (100, 233, 197*3, 1).
  """
  # Converts from (100, 233, 197) to (100, 233, 197, 1)
  inputs_op = tf.expand_dims(inputs_op, axis=3)
  target_masks_op = tf.expand_dims(target_masks_op, axis=3)
  predicted_masks_op = tf.cast(tf.expand_dims(predicted_masks_op, axis=3),
                               dtype=tf.float32)

  triplets = tf.concat([inputs_op, target_masks_op, predicted_masks_op],
                       axis=2)

  tf.summary.image("triplets", triplets[:num_images], max_outputs=num_images)
