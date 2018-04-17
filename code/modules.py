import numpy as np
import tensorflow as tf


class ConvEncoder(object):
  def __init__(self, input_shape, keep_prob, scope_name):
    self.input_shape = input_shape
    self.keep_prob = keep_prob
    self.scope_name = scope_name

  def build_graph(self, input):
    with tf.variable_scope(self.scope_name):
      conv1 = self.conv2d(input, filter_shape=[3, 3, 1, 8], scope_name="conv1")  # (232, 196, 8)
      pool1 = self.maxpool2d(conv1, scope_name="pool1")  # (116, 98, 8)
      drop1 = self.dropout(pool1, keep_prob=self.keep_prob, scope_name="drop1")
      conv2 = self.conv2d(drop1, filter_shape=[5, 5, 8, 16], scope_name="conv2")  # (116, 98, 16)
      pool2 = self.maxpool2d(conv2, scope_name="pool2")  # (58, 49, 16)
      drop2 = self.dropout(pool2, keep_prob=self.keep_prob, scope_name="drop2")
      drop2 = tf.reshape(drop2, shape=[-1, 58*49*16])  # (45472,)
      fc1 = self.fc(drop2, output_shape=1024, scope_name="fc1")
      drop3 = self.dropout(fc1, keep_prob=self.keep_prob, scope_name="drop3")
      fc2 = self.fc(drop3, output_shape=256, scope_name="fc2")
    return fc2

  def conv2d(self, input, filter_shape, scope_name, strides=[1, 1, 1, 1]):
    xavier_initializer = tf.contrib.layers.xavier_initializer
    with tf.variable_scope(scope_name):
      W = tf.get_variable(initializer=xavier_initializer(uniform=False),
                          name="W",
                          shape=filter_shape)
      b = tf.get_variable(initializer=xavier_initializer(uniform=False),
                          name="b",
                          shape=[filter_shape[3]])
      out = tf.nn.conv2d(input, W, padding="SAME", strides=strides)
      out = tf.nn.bias_add(out, b)
      out = tf.nn.relu(out)
      return out

  def maxpool2d(self, input, scope_name, pool_shape=[1, 2, 2, 1], strides=[1, 2, 2, 1]):
    with tf.variable_scope(scope_name):
      out = tf.nn.max_pool(input,
                           ksize=pool_shape,
                           strides=strides,
                           padding="SAME")
      return out

  def dropout(self, input, keep_prob, scope_name):
    with tf.variable_scope(scope_name):
      out = tf.nn.dropout(input, keep_prob)
      return out

  def fc(self, input, output_shape, scope_name):
    xavier_initializer = tf.contrib.layers.xavier_initializer
    with tf.variable_scope(scope_name):
      input_shape = input.shape[1:]
      input_shape = int(np.prod(input_shape))
      W = tf.get_variable(name="W",
                          shape=[input_shape, output_shape],
                          initializer=xavier_initializer(uniform=False))
      b = tf.get_variable(name="b",
                          shape=[output_shape],
                          initializer=xavier_initializer(uniform=False))
      input = tf.reshape(input, [-1, input_shape])
      out = tf.nn.relu(tf.add(tf.matmul(input, W), b))
      return out


class DeconvDecoder(ConvEncoder):
  def __init__(self, keep_prob, output_shape, scope_name):
    self.keep_prob = keep_prob
    self.output_shape = output_shape
    self.scope_name = scope_name

  def build_graph(self, input):
    with tf.variable_scope(self.scope_name):
      fc1 = self.fc(input, output_shape=1024, scope_name="fc1")
      drop1 = self.dropout(fc1, keep_prob=self.keep_prob, scope_name="drop1")
      fc2 = self.fc(drop1, output_shape=58*49*16, scope_name="fc2")
      drop2 = self.dropout(fc2, keep_prob=self.keep_prob, scope_name="drop2")
      drop2 = tf.reshape(drop2, shape=[-1, 58, 49, 16])
      up1 = self.upsample(drop2, scope_name="up1", factor=[2, 2])  # (116, 98, 16)
      deconv1 = self.deconv2d(up1, filter_shape=[5, 5], num_outputs=8, scope_name="deconv1")  # (116, 98, 8)
      up2 = self.upsample(deconv1, scope_name="up2", factor=[2, 2])
      deconv2 = self.deconv2d(up2, filter_shape=[3, 3], num_outputs=1, scope_name="deconv2")  # (232, 196, 1)
      return deconv2

  def deconv2d(self, input, filter_shape, num_outputs, scope_name, strides=[1, 1]):
    xavier_initializer = tf.contrib.layers.xavier_initializer
    xavier_initializer_conv2d = tf.contrib.layers.xavier_initializer_conv2d
    with tf.variable_scope(scope_name):
      out = tf.contrib.layers.conv2d_transpose(input,
                                               activation_fn=tf.nn.relu,
                                               biases_initializer=xavier_initializer(uniform=False),
                                               kernel_size=filter_shape,
                                               num_outputs=num_outputs,
                                               padding="SAME",
                                               stride=strides,
                                               weights_initializer=xavier_initializer_conv2d(uniform=False))
      return out

  def upsample(self, input, scope_name, factor=[2,2]):
    size = [int(input.shape[1] * factor[0]), int(input.shape[2] * factor[1])]
    with tf.variable_scope(scope_name):
      out = tf.image.resize_bilinear(input, size=size, align_corners=None, name=None)
      return out
