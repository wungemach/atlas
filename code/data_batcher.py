import numpy as np
import random
import time
import re

from PIL import Image


class Batch(object):
  def __init__(self, inputs_batch, target_masks_batch):
    """
    Initializes a Batch.

    Inputs:
    - inputs_batch: A numpy array with shape batch_size by input_dims that
      represents a batch of inputs.
    - target_masks_batch: A numpy array with shape batch_size by input_dims
      that represents a batch of target masks.

    Outputs:
    - None
    """
    self.inputs_batch = inputs_batch
    self.target_masks_batch = target_masks_batch
    self.batch_size = len(self.inputs_batch)


class SliceBatchGenerator(object):
  def __init__(self,
               input_path_lists,
               target_mask_path_lists,
               batch_size,
               max_num_batches=1000,
               shape=(197, 233),
               shuffle=False,
               use_fake_target_masks=False):
    """
    Inputs:
    - input_path_lists: A list of lists of Python strs that represent paths
      to input image files.
    - target_mask_path_lists: A list of lists of Python strs that represent
      paths to target mask files.
    - batch_size: A Python int that represents the batch size.
    - max_num_batches: A Python int that represents the maximum number of
      slices to refill at a time.
    - shape: A Python tuple of ints that represents the shape to resize the
      slice as (height, width).
    - shuffle: A Python bool that represents whether to shuffle the batches
      from the original order specified by {input_path_lists} and
      {target_mask_path_lists}.
    - use_fake_target_masks: A Python bool that represents whether to use
      fake target masks or not. If True, then {target_mask_path_lists} is
      ignored and all masks are all 0s. This option might be useful to sanity
      check new models before training on the real dataset.
    """
    self._input_path_lists = input_path_lists
    self._target_mask_path_lists = target_mask_path_lists
    self._batch_size = batch_size
    self._batches = []
    self._max_num_batches = max_num_batches
    self._pointer = 0
    self._order = list(range(len(self._input_path_lists)))
    self._shape = shape
    self._use_fake_target_masks = use_fake_target_masks
    if shuffle:
      random.shuffle(self._order)


  def refill_batches(self):
    """
    Refills {self._batches}.
    """
    if self._pointer >= len(self._input_path_lists):
      return

    examples = []  # A Python list of (input, target_mask) tuples

    # {start_idx} and {end_idx} are values like 2000 and 3000
    # If shuffle=True, then {self._order} is a list like [56, 720, 12, ...]
    # {path_indices} is the sublist of {self._order} that represents the
    #   current batch; in other words, the current batch of inputs will be:
    #   [self._input_path_lists[path_indices[0]],
    #    self._input_path_lists[path_indices[1]],
    #    self._input_path_lists[path_indices[2]],
    #    ...]
    # {input_path_lists} and {target_mask_path_lists} are lists of paths corresponding
    #   to the indices given by {path_indices}
    start_idx, end_idx = self._pointer, self._pointer + self._max_num_batches
    path_indices = self._order[start_idx:end_idx]
    input_path_lists = [
      self._input_path_lists[path_idx] for path_idx in path_indices]
    target_mask_path_lists = [
      self._target_mask_path_lists[path_idx] for path_idx in path_indices]
    zipped_path_lists = zip(input_path_lists, target_mask_path_lists)

    # Updates self._pointer for the next call to {self.refill_batches}
    self._pointer += self._max_num_batches

    for input_path_list, target_mask_path_list in zipped_path_lists:
      if self._use_fake_target_masks:
        input = Image.open(input_path_list[0]).convert("L")
        # Image.resize expects (width, height) order
        examples.append((
          np.asarray(input.resize(self._shape[::-1], Image.NEAREST)),
          np.zeros(self._shape)
        ))
      else:
        # Assumes {input_path_list} is a list with length 1;
        # opens input, resizes it, converts to a numpy array
        input = Image.open(input_path_list[0]).convert("L")
        input = input.resize(self._shape[::-1], Image.NEAREST)
        input = np.asarray(input)

        # Assumes {target_mask_path_list} is a list with length >= 1;
        # Merges target masks if list contains more than one path
        target_mask_list = list(map(
          lambda target_mask_path: Image.open(target_mask_path).convert("L"),
          target_mask_path_list))
        target_mask_list = list(map(
          lambda target_mask: target_mask.resize(self._shape[::-1], Image.NEAREST),
          target_mask_list))
        target_mask_list = list(map(
          lambda target_mask: (np.asarray(target_mask.resize(self._shape[::-1], Image.NEAREST)) > 1e-8) + 0.0,
          target_mask_list))
        target_mask = np.minimum(np.sum(target_mask_list, axis=0), 1.0)

        # Image.resize expects (width, height) order
        examples.append((
          input,
          # Converts all values >0 to 1s
          target_mask
        ))
      if len(examples) >= self._batch_size * self._max_num_batches:
        break

    for batch_start_idx in range(0, len(examples), self._batch_size):
      input_batch, target_mask_batch =\
        zip(*examples[batch_start_idx:batch_start_idx+self._batch_size])
      self._batches.append((input_batch, target_mask_batch))


  def get_batch(self):
    """
    Returns a generator object that yields batches, the last of which will be
    partial.
    """
    while True:
      if len(self._batches) == 0: # adds more batches
        self.refill_batches()
      if len(self._batches) == 0:
        break

      # Pops the next batch, both numpy arrays of batch_size by input_dims
      (inputs_batch, target_masks_batch) = self._batches.pop(0)

      # Wraps the numpy arrays into a Batch object
      batch = Batch(inputs_batch, target_masks_batch)

      yield batch

  def get_num_batches(self):
    """
    Returns the number of batches.
    """
    return int(len(self._input_path_lists) / self._batch_size)
