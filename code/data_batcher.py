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
               input_paths,
               target_mask_paths,
               batch_size,
               max_num_batches=1000,
               shuffle=False):
    """
    Inputs:
    - input_paths: A list of Python strs that represent pathnames to input
      image files.
    - target_mask_paths: A list of Python strs that represent pathnames to
      target mask files.
    - batch_size: A Python int that represents the batch size.
    - shuffle: A Python bool that represents whether to shuffle the batches from
      the original order specified by {input_paths} and {target_mask_paths}.
    """
    self._input_paths = input_paths
    self._target_mask_paths = target_mask_paths
    self._batch_size = batch_size
    self._batches = []
    self._max_num_batches = max_num_batches
    self._pointer = 0
    self._order = list(range(len(input_paths)))
    if shuffle:
      random.shuffle(self._order)


  def refill_batches(self):
    """
    Refills {self._batches}.
    """
    if self._pointer >= len(self._input_paths):
      return

    examples = []  # A Python list of (input, target_mask) tuples

    # {start_idx} and {end_idx} are values like 2400 and 2640
    # If shuffle=True, then {self._order} is a list like [56, 720, 12, ...]
    # {path_indices} is the current sublist of {self._order}
    # {input_paths} and {target_mask_paths} are lists of paths corresponding
    #   to the indices given by {path_indices}
    start_idx, end_idx = self._pointer, self._pointer + self._max_num_batches
    path_indices = self._order[start_idx:end_idx]
    input_paths = [self._input_paths[path_idx] for path_idx in path_indices]
    target_mask_paths = [self._target_mask_paths[path_idx] for path_idx in path_indices]
    zipped_paths = zip(input_paths, target_mask_paths)

    # Updates self._pointer for the next call to {self.refill_batches}
    self._pointer += self._max_num_batches

    for input_path, target_mask_path in zipped_paths:
      examples.append((
        np.asarray(Image.open(input_path).convert("L").resize((232, 196), Image.NEAREST)),
        np.asarray(Image.open(target_mask_path).convert("L").resize((232, 196), Image.NEAREST))
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
    return int(len(self._input_paths) / self._batch_size)
