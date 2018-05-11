import logging
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import tensorflow as tf
import time

from tqdm import tqdm

import utils
from data_batcher import SliceBatchGenerator
from modules import ConvEncoder, DeconvDecoder, UNet


class ATLASModel(object):
  def __init__(self, FLAGS):
    """
    Initializes the ATLAS model.

    Inputs:
    - FLAGS: A _FlagValuesWrapper object.
    """
    self.FLAGS = FLAGS

    var_init = tf.contrib.layers.variance_scaling_initializer
    with tf.variable_scope("ATLASModel",
                           initializer=var_init(factor=1.0, uniform=True)):
      self.add_placeholders()
      self.build_graph()
      self.add_loss()

    # Defines the trainable parameters, gradient, gradient norm, and clip by
    # gradient norm
    params = tf.trainable_variables()
    gradients = tf.gradients(self.loss, params)
    self.gradient_norm = tf.global_norm(gradients)
    clipped_gradients, _ = tf.clip_by_global_norm(gradients,
                                                  FLAGS.max_gradient_norm)
    self.param_norm = tf.global_norm(params)

    # Defines optimizer and updates; {self.updates} needs to be fetched in
    # sess.run to do a gradient update
    self.global_step = tf.Variable(0, name="global_step", trainable=False)
    opt = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)
    self.updates = opt.apply_gradients(zip(clipped_gradients, params),
                                       global_step=self.global_step)

    # Defines savers (for checkpointing) and summaries (for tensorboard)
    self.saver = tf.train.Saver(tf.global_variables(), max_to_keep=FLAGS.keep)
    self.summaries = tf.summary.merge_all()


  def add_placeholders(self):
    """
    Adds placeholders to the graph.

    Defines:
    - self.batch_size_op: A scalar placeholder Tensor that represents the
      batch size.
    - self.inputs_op: A placeholder Tensor with shape batch size by image dims
      e.g. (100, 233, 197) that represents the batch of inputs.
    - self.target_masks_op: A placeholder Tensor with shape batch size by mask
      dims e.g. (100, 233, 197) that represents the batch of target masks.
    - self.keep_prob: A scalar placeholder Tensor that represents the keep
      probability for dropout.
    """
    # Adds placeholders for inputs
    self.batch_size_op = tf.placeholder(tf.int32, shape=(), name="batch_size")

    # Defines the input dimensions, which depend on the intended input; here
    # the intended input is a single slice but volumetric inputs might require
    # 1+ additional dimensions
    self.input_dims = [self.FLAGS.slice_height, self.FLAGS.slice_width]
    self.output_dims = self.input_dims

    # Defines input and target segmentation mask according to the input dims
    self.inputs_op = tf.placeholder(tf.float32,
                                    shape=[None] + self.input_dims,
                                    name="input")
    self.target_masks_op = tf.placeholder(tf.float32,
                                          shape=[None] + self.input_dims,
                                          name="target_mask")

    # Adds a placeholder to feed in the keep probability (for dropout)
    self.keep_prob = tf.placeholder_with_default(1.0, shape=())


  def build_graph(self):
    """
    Builds the main part of the graph for the model.

    Defines:
    - self.logits_op: A Tensor of the same shape as self.inputs_op and
      self.target_masks_op e.g. (100, 233, 197) that represents the unscaled
      logits.
    - self.predicted_mask_probs_op: A Tensor of the same shape as
      self.logits_op e.g. (100, 233, 197), and passed through a sigmoid layer.
    """
    assert(self.input_dims == self.inputs_op.get_shape().as_list()[1:])
    encoder = ConvEncoder(input_shape=self.input_dims,
                          keep_prob=self.keep_prob,
                          scope_name="encoder")
    encoder_hiddens_op = encoder.build_graph(tf.expand_dims(self.inputs_op, 3))
    decoder = DeconvDecoder(keep_prob=self.keep_prob,
                            output_shape=self.input_dims,
                            scope_name="decoder")
    # Only squeezes the last dimension (do not squeeze the batch dimension)
    self.logits_op = tf.squeeze(decoder.build_graph(encoder_hiddens_op), axis=3)
    self.predicted_mask_probs_op = tf.sigmoid(self.logits_op)


  def add_loss(self):
    """
    Adds loss computation to the graph.

    Defines:
    - self.loss: A scalar Tensor that represents the loss; applies sigmoid
      cross entropy to {self.logits_op}; {self.logits_op} contains unscaled
      logits e.g. [-4.4, 1.3, -1.6, 3.5, 2.3, ...]. This particular set of
      logits incurs a high loss for {self.target_masks_op} [1, 0, 1, 0, ...]
      and low loss for [0, 1, 0, 1, 1].
    """
    with tf.variable_scope("loss"):
      # sigmoid_ce_with_logits = tf.nn.sigmoid_cross_entropy_with_logits
      # # {loss} is the same shape as {self.logits_op} and {self.target_masks_op}
      # loss = sigmoid_ce_with_logits(logits=self.logits_op,
      #                               labels=self.target_masks_op,
      #                               name="ce")

      weighted_ce_with_logits = tf.nn.weighted_cross_entropy_with_logits
      loss = weighted_ce_with_logits(logits=self.logits_op,
                                     targets=self.target_masks_op,
                                     pos_weight=100.0,
                                     name="ce")

      self.loss = tf.reduce_mean(loss)  # scalar mean across batch
      tf.summary.scalar("loss", self.loss)  # logs to TensorBoard


  def run_train_iter(self, sess, batch, summary_writer):
    """
    This performs a single training iteration: (forward-pass, loss, backprop,
    parameter update).

    Inputs:
    - sess: A TensorFlow Session object.
    - batch: A Batch object.
    - summary_writer: A SummaryWriter object for Tensorboard.

    Outputs:
    - loss: The loss (averaged across the batch) for this batch.
    - global_step: The current number of training iterations we've done.
    - param_norm: Global norm of the parameters.
    - gradient_norm: Global norm of the gradients.
    """
    # Fills the placeholders
    input_feed = {}
    input_feed[self.batch_size_op] = self.FLAGS.batch_size
    input_feed[self.inputs_op] = batch.inputs_batch
    input_feed[self.target_masks_op] = batch.target_masks_batch
    input_feed[self.keep_prob] = 1.0 - self.FLAGS.dropout  # applies dropout

    # To debug, it's often helpful to inspect variables or outputs of layers
    #   of the network. To do so, simply add tensors to {output_feed} and
    #   corresponding Python values will populate {results} e.g. suppose you
    #   wanted to inspect the output of the encoder of ATLASModel. Adding:
    #
    # "encoder_out": tf.get_default_graph().get_tensor_by_name("ATLASModel/encoder/out:0"),
    #
    # to {output_feed} causes {results["encoder_out"]} to contain numpy arrays
    # of embedding values.

    # Specifies ops to be fetched
    output_feed = {
      "updates": self.updates,
      "summaries": self.summaries,
      "loss": self.loss,
      "global_step": self.global_step,
      "param_norm": self.param_norm,
      "grad_norm": self.gradient_norm
    }

    # Runs the model
    results = sess.run(output_feed, input_feed)

    # Adds all summaries in the graph to Tensorboard
    summary_writer.add_summary(results["summaries"], results["global_step"])

    return (
      results["loss"],
      results["global_step"],
      results["param_norm"],
      results["grad_norm"]
    )


  def get_loss(self, sess, batch):
    """
    Runs a forward-pass only; gets the loss.

    Inputs:
    - sess: A TensorFlow Session object.
    - batch: A Batch object.

    Outputs:
    - loss: The loss (averaged across the batch) for this batch.
    """
    input_feed = {}
    input_feed[self.batch_size_op] = self.FLAGS.batch_size
    input_feed[self.inputs_op] = batch.inputs_batch
    input_feed[self.target_masks_op] = batch.target_masks_batch
    # keep_prob not input, so it will default to 1 i.e. no dropout

    output_feed = { "loss": self.loss }

    results = sess.run(output_feed, input_feed)

    return results["loss"]


  def get_predicted_mask_probs(self, sess, batch):
    """
    Runs a forward-pass only; gets the probability distributions for the
    predicted masks i.e. sigmoid({self.logits_op}).

    Inputs:
    - sess: A TensorFlow Session object.
    - batch: A Batch object.

    Outputs:
    - predicted_mask_probs: A numpy array of the shape self.FLAGS.batch_size by
      self.output_dims.
    """
    input_feed = {}
    input_feed[self.batch_size_op] = self.FLAGS.batch_size
    input_feed[self.inputs_op] = batch.inputs_batch
    # keep_prob not input, so it will default to 1 i.e. no dropout

    output_feed = { "predicted_mask_probs": self.predicted_mask_probs_op }
    results = sess.run(output_feed, input_feed)
    return results["predicted_mask_probs"]


  def get_predicted_masks(self, sess, batch):
    """
    Runs a forward-pass only; gets the predicted masks.

    Inputs:
    - sess: A TensorFlow Session object.
    - batch: A Batch object.

    Outputs:
    - predicted_masks: A numpy array of the shape self.FLAGS.batch_size by
      self.output_dims.
    """
    predicted_mask_probs = self.get_predicted_mask_probs(sess, batch)
    predicted_masks = predicted_mask_probs > 0.5
    return predicted_masks


  def get_dev_loss(self,
                   sess,
                   dev_input_paths,
                   dev_target_mask_paths,
                   num_samples=None):
    """
    Get loss for entire dev set.

    Inputs:
    - sess: A TensorFlow Session object.
    - dev_input_paths: A list of Python strs that represent pathnames to input
      image files in the dev set.
    - dev_target_mask_paths: A list of Python strs that represent pathnames to
      input target mask files in the dev set.
    - num_samples: A Python int or None. If None, then evaluates on the entire
      dev set.

    Outputs:
    - dev_loss: A Python float that represents the average loss across the dev
      set.
    """
    logging.info("Calculating dev loss...")
    tic = time.time()

    loss_per_batch, batch_sizes = [], []

    sbg = SliceBatchGenerator(dev_input_paths,
                              dev_target_mask_paths,
                              self.FLAGS.batch_size,
                              num_samples=num_samples,
                              shape=(self.FLAGS.slice_height,
                                     self.FLAGS.slice_width),
                              use_fake_target_masks=self.FLAGS.use_fake_target_masks)
    # Iterates over dev set batches
    for batch in sbg.get_batch():
      # Gets loss for this batch
      loss = self.get_loss(sess, batch)
      cur_batch_size = batch.batch_size
      loss_per_batch.append(loss * cur_batch_size)
      batch_sizes.append(cur_batch_size)

    # Calculates average loss
    total_num_examples = sum(batch_sizes)

    # Overall loss is total loss divided by total number of examples
    dev_loss = sum(loss_per_batch) / float(total_num_examples)

    toc = time.time()
    logging.info(f"Calculating dev loss took {toc-tic} sec.")
    return dev_loss


  def calculate_dice_coefficient(self,
                                 sess,
                                 input_paths,
                                 target_mask_paths,
                                 dataset,
                                 num_samples=100,
                                 plot=False,
                                 print_to_screen=False):
    """
    Samples from the provided dataset and, for each sample, calculates the
    dice-coefficient score. Outputs the average dice coefficient score for all
    samples. Optionally plots examples.

    Inputs:
    - sess: A TensorFlow Session object.
    - input_paths: A list of Python strs that represent pathnames to input
      image files.
    - target_mask_paths: A list of Python strs that represent pathnames to
      target mask files.
    - dataset: A Python str that represents the dataset being tested. Options:
      {train,dev}. Just for logging purposes.
    - num_samples: A Python int that represents the number of samples to test.
      If num_samples=None, then test whole dataset.
    - plot: A Python bool. If True, plots each example to screen.

    Outputs:
    - dice_coefficient: A Python float that represents the average dice
      coefficient across the sampled examples.
    """
    logging.info(f"Calculating dice coefficient for {num_samples} examples "
                 f"from {dataset}...")
    tic = time.time()

    dice_coefficient_total = 0.
    num_examples = 0

    sbg = SliceBatchGenerator(input_paths,
                              target_mask_paths,
                              self.FLAGS.batch_size,
                              shape=(self.FLAGS.slice_height,
                                     self.FLAGS.slice_width),
                              use_fake_target_masks=self.FLAGS.use_fake_target_masks)
    for batch in sbg.get_batch():
      predicted_masks = self.get_predicted_masks(sess, batch)

      zipped_masks = zip(predicted_masks,
                         batch.target_masks_batch,
                         batch.input_paths_batch,
                         batch.target_mask_path_lists_batch)
      for idx, (predicted_mask,
                target_mask,
                input_path,
                target_mask_path_list) in enumerate(zipped_masks):
        dice_coefficient = utils.dice_coefficient(predicted_mask, target_mask)
        if dice_coefficient >= 0.0:
          dice_coefficient_total += dice_coefficient
          num_examples += 1

          if print_to_screen:
            # Whee! We predicted at least one lesion pixel!
            logging.info(f"Dice coefficient of valid example {num_examples}: "
                         f"{dice_coefficient}")
          if plot:
            f, axarr = plt.subplots(1, 2)
            f.suptitle(input_path)
            axarr[0].imshow(predicted_mask)
            axarr[0].set_title("Predicted")
            axarr[1].imshow(target_mask)
            axarr[1].set_title("Target")
            examples_dir = os.path.join(self.FLAGS.train_dir, "examples")
            if not os.path.exists(examples_dir):
              os.makedirs(examples_dir)
            f.savefig(os.path.join(examples_dir, str(num_examples).zfill(4)))

        if num_samples != None and num_examples >= num_samples:
          break

      if num_samples != None and num_examples >= num_samples:
        break

    dice_coefficient_mean = dice_coefficient_total / num_examples

    toc = time.time()
    logging.info(f"Calculating dice coefficient took {toc-tic} sec.")
    return dice_coefficient_mean


  def train(self,
            sess,
            train_input_paths,
            train_target_mask_paths,
            dev_input_paths,
            dev_target_mask_paths):
    """
    Defines the training loop.

    Inputs:
    - sess: A TensorFlow Session object.
    - {train,dev}_{input_paths,target_mask_paths}: A list of Python strs
      that represent pathnames to input image files and target mask files.
    """
    params = tf.trainable_variables()
    num_params = sum(map(lambda t: np.prod(tf.shape(t.value()).eval()), params))

    # We will keep track of exponentially-smoothed loss
    exp_loss = None

    # Checkpoint management.
    # We keep one latest checkpoint, and one best checkpoint (early stopping)
    checkpoint_path = os.path.join(self.FLAGS.train_dir, "qa.ckpt")
    best_dev_dice_coefficient = None

    # for TensorBoard
    summary_writer = tf.summary.FileWriter(self.FLAGS.train_dir, sess.graph)

    epoch = 0
    num_epochs = self.FLAGS.num_epochs
    while num_epochs == None or epoch < num_epochs:
      epoch += 1

      # Loops over batches
      sbg = SliceBatchGenerator(train_input_paths,
                                train_target_mask_paths,
                                self.FLAGS.batch_size,
                                shape=(self.FLAGS.slice_height,
                                       self.FLAGS.slice_width),
                                use_fake_target_masks=self.FLAGS.use_fake_target_masks)
      num_epochs_str = str(num_epochs) if num_epochs != None else "indefinite"
      for batch in tqdm(sbg.get_batch(),
                        desc=f"Epoch {epoch}/{num_epochs_str}",
                        total=sbg.get_num_batches()):
        # Runs training iteration
        loss, global_step, param_norm, grad_norm =\
          self.run_train_iter(sess, batch, summary_writer)

        # Updates exponentially-smoothed loss
        if not exp_loss:  # first iter
          exp_loss = loss
        else:
          exp_loss = 0.99 * exp_loss + 0.01 * loss

        # Sometimes prints info
        if global_step % self.FLAGS.print_every == 0:
          logging.info(
            f"epoch {epoch}, "
            f"global_step {global_step}, "
            f"loss {loss}, "
            f"exp_loss {exp_loss}, "
            f"grad norm {grad_norm}, "
            f"param norm {param_norm}")

        # Sometimes saves model
        if (global_step % self.FLAGS.save_every == 0
            or global_step == sbg.get_num_batches()):
          self.saver.save(sess, checkpoint_path, global_step=global_step)

        # Sometimes evaluates model on dev loss, train F1/EM and dev F1/EM
        if global_step % self.FLAGS.eval_every == 0:
          # Logs loss for entire dev set to TensorBoard
          dev_loss = self.get_dev_loss(sess,
                                       dev_input_paths,
                                       dev_target_mask_paths,
                                       self.FLAGS.dev_num_samples)
          logging.info(f"epoch {epoch}, "
                       f"global_step {global_step}, "
                       f"dev_loss {dev_loss}")
          write_summary(dev_loss, "dev/loss", summary_writer, global_step)

          # Logs dice coefficient on train set to TensorBoard
          train_dice = self.calculate_dice_coefficient(
            sess, train_input_paths, train_target_mask_paths, "train")
          logging.info(f"epoch {epoch}, "
                       f"global_step {global_step}, "
                       f"train dice_coefficient: {train_dice}")
          write_summary(train_dice, "train/dice", summary_writer, global_step)

          # Logs dice coefficient on dev set to TensorBoard
          dev_dice = self.calculate_dice_coefficient(
            sess, dev_input_paths, dev_target_mask_paths, "dev")
          logging.info(f"epoch {epoch}, "
                       f"global_step {global_step}, "
                       f"dev dice_coefficient: {dev_dice}")
          write_summary(dev_dice, "dev/dice", summary_writer, global_step)
      # end for batch in sbg.get_batch
    # end while num_epochs == 0 or epoch < num_epochs
    sys.stdout.flush()


class ZeroATLASModel(ATLASModel):
  def __init__(self, FLAGS):
    """
    Initializes the Zero ATLAS model, which predicts 0 for the entire mask
    no matter what, which performs well when --use_fake_target_masks.

    Inputs:
    - FLAGS: A _FlagValuesWrapper object passed in from main.py.
    """
    super().__init__(FLAGS)


  def build_graph(self):
    """
    Sets {self.logits_op} to a matrix entirely of a small constant.
    """
    # -18.420680734 produces a sigmoid-ce loss of ~10^-8
    c = tf.get_variable(initializer=tf.constant_initializer(-18.420680734),
                        name="c",
                        shape=())
    self.logits_op = tf.ones(shape=[self.FLAGS.batch_size] + self.input_dims,
                             dtype=tf.float32) * c
    self.predicted_mask_probs_op = tf.sigmoid(self.logits_op)


class UNetATLASModel(ATLASModel):
  def __init__(self, FLAGS):
    """
    Initializes the U-Net ATLAS model, which predicts 0 for the entire mask
    no matter what, which performs well when --use_fake_target_masks.

    Inputs:
    - FLAGS: A _FlagValuesWrapper object passed in from main.py.
    """
    super().__init__(FLAGS)

  def build_graph(self):
    assert(self.input_dims == self.inputs_op.get_shape().as_list()[1:])
    unet = UNet(input_shape=self.input_dims,
                keep_prob=self.keep_prob,
                output_shape=self.input_dims,
                scope_name="unet")
    self.logits_op = tf.squeeze(
      unet.build_graph(tf.expand_dims(self.inputs_op, 3)), axis=3)

    self.predicted_mask_probs_op = tf.sigmoid(self.logits_op)


def write_summary(value, tag, summary_writer, global_step):
  """Writes a single summary value to TensorBoard."""
  summary = tf.Summary()
  summary.value.add(tag=tag, simple_value=value)
  summary_writer.add_summary(summary, global_step)
