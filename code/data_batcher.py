import numpy as np
import random
import time
import re

from PIL import Image


class Batch(object):
  def __init__(self,
               inputs_batch,
               target_masks_batch,
               input_paths_batch,
               target_mask_path_lists_batch):
    """
    Initializes a Batch.

    Inputs:
    - inputs_batch: A numpy array with shape batch_size by input_dims that
      represents a batch of inputs.
    - target_masks_batch: A numpy array with shape batch_size by input_dims
      that represents a batch of target masks.
    - input_paths_batch: A tuple of Python strs that represents a batch of
      input paths.
    - target_mask_path_lists_batch: A tuple of lists of Python strs that
      represents a batch of target mask path lists, where each list provides
      the target mask paths corresponding to the input paths.

    Outputs:
    - None
    """
    self.inputs_batch = inputs_batch
    self.target_masks_batch = target_masks_batch
    self.batch_size = len(self.inputs_batch)
    self.input_paths_batch = input_paths_batch
    self.target_mask_path_lists_batch = target_mask_path_lists_batch


class SliceBatchGenerator(object):
  def __init__(self,
               input_path_lists,
               target_mask_path_lists,
               batch_size,
               max_num_refill_batches=1000,
               num_samples=None,
               shape=(197, 233),
               shuffle=False,
               use_fake_target_masks=False):
    """
    Initializes a SliceBatchGenerator.

    Inputs:
    - input_path_lists: A list of lists of Python strs that represent paths
      to input image files.
    - target_mask_path_lists: A list of list of lists of Python strs that
      represent paths to target mask files.
    - batch_size: A Python int that represents the batch size.
    - max_num_refill_batches: A Python int that represents the maximum number
      of batches of slices to refill at a time.
    - num_samples: A Python int or None. If None, then uses the entire
      {input_path_lists} and {target_mask_path_lists}.
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
    self._max_num_refill_batches = max_num_refill_batches
    self._num_samples = num_samples
    if self._num_samples != None:
      self._input_path_lists = self._input_path_lists[:self._num_samples]
      self._target_mask_path_lists = self._target_mask_path_lists[:self._num_samples]
    self._pointer = 0
    self._order = list(range(len(self._input_path_lists)))

    # When the batch_size does not even divide the number of input paths,
    # fill the last batch with randomly selected paths
    num_others = self._batch_size - (len(self._order) % self._batch_size)
    self._order += random.choices(self._order, k=num_others)

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
    start_idx, end_idx = self._pointer, self._pointer + self._max_num_refill_batches
    path_indices = self._order[start_idx:end_idx]
    input_path_lists = [
      self._input_path_lists[path_idx] for path_idx in path_indices]
    target_mask_path_lists = [
      self._target_mask_path_lists[path_idx] for path_idx in path_indices]
    zipped_path_lists = zip(input_path_lists, target_mask_path_lists)

    # Updates self._pointer for the next call to {self.refill_batches}
    self._pointer += self._max_num_refill_batches

    for input_path_list, target_mask_path_list in zipped_path_lists:
      if self._use_fake_target_masks:
        input = Image.open(input_path_list[0]).convert("L")
        # Image.resize expects (width, height) order
        examples.append((
          np.asarray(input.resize(self._shape[::-1], Image.NEAREST)),
          np.zeros(self._shape),
          input_path_list[0],
          "fake_target_mask"
        ))
      else:
        # Assumes {input_path_list} is a list with length 1;
        # opens input, resizes it, converts to a numpy array
        input = Image.open(input_path_list[0]).convert("L")
        input = input.resize(self._shape[::-1], Image.NEAREST)
        input = np.asarray(input) / 255.0

        # Assumes {target_mask_path_list} is a list of lists, where the outer
        # list has length 1 and the inner list has length >= 1;
        # Merges target masks if list contains more than one path
        target_mask_list = list(map(
          lambda target_mask_path: Image.open(target_mask_path).convert("L"),
          target_mask_path_list[0]))
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
          target_mask,
          input_path_list[0],
          target_mask_path_list[0]
        ))
      if len(examples) >= self._batch_size * self._max_num_refill_batches:
        break

    for batch_start_idx in range(0, len(examples), self._batch_size):
      (inputs_batch, target_masks_batch, input_paths_batch,
       target_mask_path_lists_batch) =\
         zip(*examples[batch_start_idx:batch_start_idx+self._batch_size])
      self._batches.append((np.asarray(inputs_batch),
                            np.asarray(target_masks_batch),
                            input_paths_batch,
                            target_mask_path_lists_batch))


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

      # Pops the next batch, a tuple of four items; the first two are numpy
      # arrays {inputs_batch} and {target_mask_batch} of batch_size by
      # input_dims, the last two are tuples of paths.
      batch = self._batches.pop(0)

      # Wraps the numpy arrays into a Batch object
      batch = Batch(*batch)

      yield batch


  def get_num_batches(self):
    """
    Returns the number of batches.
    """
    # The -1 then +1 accounts for the remainder batch.
    return int((len(self._input_path_lists) - 1) / self._batch_size) + 1
