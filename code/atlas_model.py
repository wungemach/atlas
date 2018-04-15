import logging
import os
import sys
import time

import numpy as np
import tensorflow as tf

import utils

from evaluate import exact_match_score, f1_score
from data_batcher import SliceBatchGenerator

logging.basicConfig(level=logging.INFO)


class ATLASModel(object):
  def __init__(self, FLAGS):
    """
    Initializes the ATLAS model.

    Inputs:
    - FLAGS: the flags passed in from main.py
    """
    self.FLAGS = FLAGS

    with tf.variable_scope("ATLASModel", initializer=tf.contrib.layers.variance_scaling_initializer(factor=1.0, uniform=True)):
      self.add_placeholders()
      self.build_graph()
      self.add_loss()

    # Define trainable parameters, gradient, gradient norm, and clip by
    # gradient norm
    params = tf.trainable_variables()
    gradients = tf.gradients(self.loss, params)
    self.gradient_norm = tf.global_norm(gradients)
    clipped_gradients, _ = tf.clip_by_global_norm(gradients,
                                                  FLAGS.max_gradient_norm)
    self.param_norm = tf.global_norm(params)

    # Define optimizer and updates
    # (updates is what you need to fetch in session.run to do a gradient update)
    self.global_step = tf.Variable(0, name="global_step", trainable=False)
    opt = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)
    self.updates = opt.apply_gradients(zip(clipped_gradients, params),
                                       global_step=self.global_step)

    # Define savers (for checkpointing) and summaries (for tensorboard)
    self.saver = tf.train.Saver(tf.global_variables(), max_to_keep=FLAGS.keep)
    self.summaries = tf.summary.merge_all()


  def add_placeholders(self):
    """
    Adds placeholders to the graph.
    """
    # Add placeholders for inputs.
    self.batch_size_op = tf.placeholder(tf.int32, shape=(), name="batch_size")

    # Defines the input dimensions, which depend on the intended input; here
    # the intended input is a single slice but volumetric inputs might require
    # 1+ additional dimensions.
    self.input_dims = [self.FLAGS.image_height, self.FLAGS.image_width]
    self.output_dims = self.input_dims

    # Defines input and target segmentation mask according to the input dims
    self.input_op = tf.placeholder(
      tf.int32, shape=[self.batch_size_op] + self.input_dims)
    self.target_mask_op = tf.placeholder(
      tf.int32, shape=[self.batch_size_op] + self.input_dims)

    # Adds a placeholder to feed in the keep probability (for dropout).
    self.keep_prob = tf.placeholder_with_default(1.0, shape=())


  def build_graph(self):
    """
    Builds the main part of the graph for the model.

    Defines:
    - self.logits_op: A Tensor of the same shape as self.input_op and
      self.target_mask_op.
    - self.predicted_mask_probs_op: A Tensor of the same shape as
      self.logits_op, passed through a sigmoid layer.
    """
    # from models.official.resnet.resnet_model import Model as ResNet
    # encoder = ResNet(resnet_size=self.FLAGS.resnet_size,
    #                  bottleneck=self.FLAGS.resnet_size < 50,
    #                  num_classes=1,  # TODO: replace
    #                  num_filters=64,
    #                  kernel_size=7,
    #                  conv_stride=2,
    #                  first_pool_size=3,
    #                  first_pool_stride=2,
    #                  second_pool_size=7,
    #                  second_pool_stride=1,
    #                  block_sizes=utils.get_block_sizes(resnet_size),
    #                  block_strides=[1, 2, 2, 2],
    #                  final_size=512 if self.FLAGS.resnet_size < 50 else 2048,
    #                  version=2,  # resnet_model.DEFAULT_VERSION
    #                  data_format=None,
    #                  dtype=tf.float32)  # resnet_model.DEFAULT_DTYPE
    self.logits_op = tf.zeros_like(self.input_op)
    self.predicted_mask_probs_op = tf.sigmoid(self.logits_op)


  def add_loss(self):
    """
    Adds loss computation to the graph.

    Defines:
    - self.loss: A scalar Tensor.
    """
    with tf.variable_scope("loss"):
      loss = tf.nn.sigmoid_cross_entropy_with_logits(
        logits=self.logits_op,
        labels=self.target_mask_op)
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
    - global_step: The current number of training iterations we've done
    - param_norm: Global norm of the parameters
    - gradient_norm: Global norm of the gradients
    """
    # Match up our input data with the placeholders
    input_feed = {}
    input_feed[self.batch_size_op] = self.FLAGS.batch_size
    input_feed[self.input_op] = batch.input
    input_feed[self.target_mask_op] = batch.target_mask
    input_feed[self.keep_prob] = 1.0 - self.FLAGS.dropout  # applies dropout

    # output_feed contains the things we want to fetch.
    output_feed = {
      "updates": self.updates,
      "summaries": self.summaries,
      "loss": self.loss,
      "global_step": self.global_step,
      "param_norm": self.param_norm,
      "grad_norm": self.gradient_norm
    }

    # Run the model
    results = sess.run(output_feed, input_feed)

    # All summaries in the graph are added to Tensorboard
    summary_writer.add_summary(results["summaries"], results["global_step"])

    return results["loss"], results["global_step"], results["param_norm"], results["grad_norm"]


  def get_loss(self, sess, batch):
    """
    Runs forward-pass only; get loss.

    Inputs:
    - sess: A TensorFlow Session object.
    - batch: A Batch object.

    Outputs:
    - loss: The loss (averaged across the batch) for this batch.
    """
    input_feed = {}
    input_feed[self.batch_size_op] = self.FLAGS.batch_size
    input_feed[self.input_op] = batch.input
    input_feed[self.target_mask_op] = batch.target_mask
    # keep_prob not input, so it will default to 1 i.e. no dropout

    output_feed = { "loss": self.loss }

    results = sess.run(output_feed, input_feed)

    return results["loss"]


  def get_predicted_mask_probs(self, sess, batch):
    """
    Run forward-pass only; get probability distributions for the predicted
    masks.

    Inputs:
    - sess: A TensorFlow Session object.
    - batch: A Batch object.

    Outputs:
    - predicted_mask_probs: A numpy array of the shape self.FLAGS.batch_size by
      self.output_dims.
    """
    input_feed = {}
    input_feed[self.batch_size_op] = self.FLAGS.batch_size
    input_feed[self.input_op] = batch.input
    # keep_prob not input, so it will default to 1 i.e. no dropout

    output_feed = { "predicted_mask_probs": self.predicted_mask_probs_op }
    results = sess.run(output_feed, input_feed)
    return results["predicted_mask_probs"]


  def get_predicted_masks(self, sess, batch):
    """
    Run forward-pass only; get probability distributions for the predicted
    masks.

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


  def get_dev_loss(self, sess, dev_input_paths, dev_target_mask_paths):
    """
    Get loss for entire dev set.

    Inputs:
    - sess: A TensorFlow Session object.

    Outputs:
    - dev_loss: A Python float that represents the average loss across the dev
      set.
    """
    loss_per_batch, batch_sizes = [], []

    sbg = SliceBatchGenerator(dev_input_paths,
                              dev_target_mask_paths
                              self.FLAGS.batch_size)
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

    return dev_loss


  def calculate_dice_coefficient(self,
                                 sess
                                 input_paths,
                                 target_mask_paths,
                                 dataset,
                                 num_samples=100,
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
    dice_coefficient_total = 0.
    num_examples = 0

    sbg = SliceBatchGenerator(input_paths,
                              target_mask_paths
                              self.FLAGS.batch_size)
    for batch in sbg.get_batch():
      predicted_masks = self.get_predicted_masks(sess, batch)

      # Convert the start and end positions to lists length batch_size
      pred_start_pos = pred_start_pos.tolist() # list length batch_size
      pred_end_pos = pred_end_pos.tolist() # list length batch_size

      zipped_masks = zip(predicted_masks, batch.target_masks)
      for idx, (predicted_mask, target_mask) in enumerate(zipped_masks):
        num_examples += 1

        dice_coefficient_total += \
          utils.dice_coefficient(predicted_mask, target_mask)

        # Optionally plot
        if plot:
          pass  # TODO: later

        if num_samples != None and num_examples >= num_samples:
          break

      if num_samples != None and num_examples >= num_samples:
        break

    dice_coefficient_mean = dice_coefficient_total / num_examples
    return dice_coefficient_mean


  def train(self,
            sess,
            train_input_paths,
            train_target_mask_paths,
            dev_input_paths,
            dev_target_mask_paths):
    """
    Defines training loop.

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
    summary_writer = tf.summary.FileWriter(self.FLAGS.train_dir, session.graph)

    epoch = 0
    while self.FLAGS.num_epochs == 0 or epoch < self.FLAGS.num_epochs:
      epoch += 1

      # Loops over batches
      sbg = SliceBatchGenerator(train_input_paths,
                                train_target_mask_paths
                                self.FLAGS.batch_size)
      for batch in sbg.get_batch():
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
            f"global_step {global_step']}, "
            f"loss {loss}, "
            f"exp_loss {exp_loss}, "
            f"grad norm {grad_norm}, "
            f"param norm {param_norm}")

        # Sometimes saves model
        if global_step % self.FLAGS.save_every == 0:
          self.saver.save(sess, checkpoint_path, global_step=global_step)

        # Sometimes evaluates model on dev loss, train F1/EM and dev F1/EM
        if global_step % self.FLAGS.eval_every == 0:
          # Logs loss for entire dev set to TensorBoard
          dev_loss = self.get_dev_loss(
            sess, dev_input_paths, dev_target_mask_paths)
          logging.info(
            f"epoch {epoch}, "
            f"global_step {global_step}, "
            f"dev_loss {dev_loss}")
          write_summary(dev_loss, "dev/loss", summary_writer, global_step)

          # Logs dice coefficient on train set to TensorBoard
          train_dice = self.calculate_dice_coefficient(
            sess, train_input_paths, train_target_mask_paths, "train", num_samples=1000)
          logging.info(
            f"epoch {epoch}, "
            f"global_step {global_step}, "
            f"train dice_coefficient: {train_dice}")
          write_summary(train_dice, "train/dice", summary_writer, global_step)

          # Logs dice coefficient on dev set to TensorBoard
          dev_dice = self.calculate_dice_coefficient(
            sess, dev_input_paths, dev_target_mask_paths, "dev")
          logging.info(
            f"epoch {epoch}, "
            f"global_step {global_step}, "
            f"dev dice_coefficient: {dev_dice}")
          write_summary(dev_dice, "dev/dice", summary_writer, global_step)
      # end for batch in sbg.get_batch
    # end while self.FLAGS.num_epochs == 0 or epoch < self.FLAGS.num_epochs
    sys.stdout.flush()


def write_summary(value, tag, summary_writer, global_step):
  """Writes a single summary value to tensorboard"""
  summary = tf.Summary()
  summary.value.add(tag=tag, simple_value=value)
  summary_writer.add_summary(summary, global_step)
