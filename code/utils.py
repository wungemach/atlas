import numpy as np

def dice_coefficient(predicted_mask, target_mask):
  tp = np.sum(np.logical_and(predicted_mask, target_mask))
  fp_fn = np.sum(np.logical_xor(predicted_mask, target_mask))
  if (2 * tp + fp_fn) == 0: return 0.0  # mask is entirely 0
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
