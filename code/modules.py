import numpy as np
import tensorflow as tf


class NeuralNetwork(object):
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
      return out

  def conv2d_relu(self, input, filter_shape, scope_name, strides=[1, 1, 1, 1], padding="SAME"):
    xavier_initializer = tf.contrib.layers.xavier_initializer
    with tf.variable_scope(scope_name):
      W = tf.get_variable(initializer=xavier_initializer(uniform=False),
                          name="W",
                          shape=filter_shape)
      b = tf.get_variable(initializer=xavier_initializer(uniform=False),
                          name="b",
                          shape=[filter_shape[3]])
      out = tf.nn.conv2d(input, W, padding=padding, strides=strides)
      out = tf.nn.bias_add(out, b)
      out = tf.nn.relu(out, name="out")
      return out

  def maxpool2d(self, input, scope_name, pool_shape=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME"):
    with tf.variable_scope(scope_name):
      out = tf.nn.max_pool(input,
                           ksize=pool_shape,
                           name="out",
                           padding=padding,
                           strides=strides)
      return out

  def dropout(self, input, keep_prob, scope_name):
    with tf.variable_scope(scope_name):
      out = tf.nn.dropout(input, keep_prob, name="out")
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
      # out = tf.nn.relu(tf.add(tf.matmul(input, W), b), name="out")
      out = tf.add(tf.matmul(input, W), b, name="out")
      return out

  def deconv2d(self, input, filter_shape, num_outputs, scope_name, strides=[1, 1]):
    xavier_initializer = tf.contrib.layers.xavier_initializer
    xavier_initializer_conv2d = tf.contrib.layers.xavier_initializer_conv2d
    with tf.variable_scope(scope_name):
      out = tf.contrib.layers.conv2d_transpose(input,
                                               # activation_fn=tf.nn.relu,
                                               activation_fn=None,
                                               biases_initializer=xavier_initializer(uniform=False),
                                               kernel_size=filter_shape,
                                               num_outputs=num_outputs,
                                               padding="SAME",
                                               stride=strides,
                                               weights_initializer=xavier_initializer_conv2d(uniform=False))
      out = tf.identity(out, name="out")
      return out

  def upsample(self, input, scope_name, factor=[2, 2]):
    size = [int(input.shape[1] * factor[0]), int(input.shape[2] * factor[1])]
    with tf.variable_scope(scope_name):
      out = tf.image.resize_bilinear(input, size=size, align_corners=None, name="out")
      return out


class ConvEncoder(NeuralNetwork):
  def __init__(self, input_shape, keep_prob, scope_name="encoder"):
    self.input_shape = input_shape
    self.keep_prob = keep_prob
    self.scope_name = scope_name

  def build_graph(self, input):
    with tf.variable_scope(self.scope_name):
      conv1 = self.conv2d_relu(input, filter_shape=[3, 3, 1, 8], scope_name="conv1")  # (232, 196, 8)
      pool1 = self.maxpool2d(conv1, scope_name="pool1")  # (116, 98, 8)
      drop1 = self.dropout(pool1, keep_prob=self.keep_prob, scope_name="drop1")
      conv2 = self.conv2d_relu(drop1, filter_shape=[5, 5, 8, 16], scope_name="conv2")  # (116, 98, 16)
      pool2 = self.maxpool2d(conv2, scope_name="pool2")  # (58, 49, 16)
      drop2 = self.dropout(pool2, keep_prob=self.keep_prob, scope_name="drop2")
      drop2 = tf.reshape(drop2, shape=[-1, 58*49*16])  # (45472,)
      fc1 = self.fc(drop2, output_shape=1024, scope_name="fc1")
      drop3 = self.dropout(fc1, keep_prob=self.keep_prob, scope_name="drop3")
      fc2 = self.fc(drop3, output_shape=256, scope_name="fc2")
      out = tf.identity(fc2, name="out")

    return out


class DeconvDecoder(NeuralNetwork):
  def __init__(self, keep_prob, output_shape, scope_name="decoder"):
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
      out = tf.identity(deconv2, name="out")

    return out


class UNet(NeuralNetwork):
  def __init__(self, input_shape, keep_prob, output_shape, scope_name="unet"):
    self.input_shape = input_shape
    self.keep_prob = keep_prob
    self.output_shape = output_shape
    self.scope_name = scope_name

  def build_graph(self, input):
    with tf.variable_scope(self.scope_name):
      # Conv
      conv1 = self.conv2d_relu(input, filter_shape=[3, 3, 1, 64], scope_name="conv1")  # (b, 232, 196, 64)
      drop1 = self.dropout(conv1, keep_prob=self.keep_prob, scope_name="drop1")
      conv2 = self.conv2d_relu(drop1, filter_shape=[3, 3, 64, 64], scope_name="conv2")  # (b, 232, 196, 64)
      drop2 = self.dropout(conv2, keep_prob=self.keep_prob, scope_name="drop2")

      pool1 = self.maxpool2d(drop2, scope_name="pool1")  # (b, 116, 98, 64)
      conv3 = self.conv2d_relu(pool1, filter_shape=[3, 3, 64, 128], scope_name="conv3")  # (b, 116, 98, 128)
      drop3 = self.dropout(conv3, keep_prob=self.keep_prob, scope_name="drop3")
      conv4 = self.conv2d_relu(drop3, filter_shape=[3, 3, 128, 128], scope_name="conv4")  # (b, 116, 98, 128)
      drop4 = self.dropout(conv4, keep_prob=self.keep_prob, scope_name="drop4")

      pool2 = self.maxpool2d(conv4, scope_name="pool2")  # (b, 58, 49, 128)
      conv5 = self.conv2d_relu(pool2, filter_shape=[3, 3, 128, 256], scope_name="conv5")  # (b, 58, 49, 256)
      drop5 = self.dropout(conv5, keep_prob=self.keep_prob, scope_name="drop5")
      conv6 = self.conv2d_relu(drop5, filter_shape=[3, 3, 256, 256], scope_name="conv6")  # (b, 58, 49, 256)
      drop6 = self.dropout(conv6, keep_prob=self.keep_prob, scope_name="drop6")

      # Deconv
      up1 = self.upsample(drop6, scope_name="up1", factor=[2, 2])  # (b, 116, 98, 256)
      deconv1 = self.deconv2d(up1, filter_shape=[2, 2], num_outputs=128, scope_name="deconv1")  # (b, 116, 98, 128)
      concat1 = tf.concat([drop4, deconv1], axis=3)  # (b, 116, 98, 256)
      conv7 = self.conv2d_relu(concat1, filter_shape=[3, 3, 256, 128], scope_name="conv7")  # (b, 116, 98, 128)
      drop7 = self.dropout(conv7, keep_prob=self.keep_prob, scope_name="drop7")
      conv8 = self.conv2d_relu(drop7, filter_shape=[3, 3, 128, 128], scope_name="conv8")  # (b, 116, 98, 128)
      drop8 = self.dropout(conv8, keep_prob=self.keep_prob, scope_name="drop8")

      up2 = self.upsample(drop8, scope_name="up2", factor=[2, 2])  # (b, 232, 196, 128)
      deconv2 = self.deconv2d(up2, filter_shape=[2, 2], num_outputs=64, scope_name="deconv2")  # (b, 232, 196, 64)
      concat2 = tf.concat([drop2, deconv2], axis=3)  # (b, 232, 196, 128)
      conv9 = self.conv2d_relu(concat2, filter_shape=[3, 3, 128, 64], scope_name="conv9")  # (b, 232, 196, 64)
      drop9 = self.dropout(conv9, keep_prob=self.keep_prob, scope_name="drop9")
      conv10 = self.conv2d_relu(drop9, filter_shape=[3, 3, 64, 64], scope_name="conv10")  # (b, 232, 196, 64)
      drop10 = self.dropout(conv10, keep_prob=self.keep_prob, scope_name="drop10")

      conv11 = self.conv2d(drop10, filter_shape=[1, 1, 64, 1], scope_name="conv11")  # (b, 232, 196, 1)
      out = tf.identity(conv11, name="out")

    return out

class DualNet(NeuralNetwork):
  def __init__(self, input_shape, keep_prob, output_shape, scope_name="unet"):
    self.input_shape = input_shape
    self.keep_prob = keep_prob
    self.output_shape = output_shape
    self.scope_name = scope_name

  def build_graph(self, input):

    '''
      - Implement the dual pathway architecture with green_box centered ~100x100, blue_box entire image
      - Spacial dims when you concatenate should be ~50x50 or maybe ~25x25
      - Deconv only on blue_path to get to correct dimensions before concatenating
      - End with 1x1 convolutions for fulling connected layer
      - Run on a couple images with lesions within the green_box
    '''
    with tf.variable_scope(self.scope_name):


      # ----- Top path -----

      # Take a crop out of the middle of the input
      green_box = tf.slice(input, [0, 103, 93, 0], [-1, 50, 50, -1], name="crop1") # (b, 50, 50, 1)

      # Initial convolutions
      conv1_top = self.conv2d_relu(green_box, filter_shape=[3, 3, 1, 30], scope_name="conv1_top", padding="VALID")  # (b, 48, 48, 30)
      drop1_top = self.dropout(conv1_top, keep_prob=self.keep_prob, scope_name="drop1_top")
      conv2_top = self.conv2d_relu(drop1_top, filter_shape=[3, 3, 30, 30], scope_name="conv2_top", padding="VALID")  # (b, 46, 46, 30)
      drop2_top = self.dropout(conv2_top, keep_prob=self.keep_prob, scope_name="drop2_top")

      # Final convolutions
      conv3_top = self.conv2d_relu(drop2_top, filter_shape=[3, 3, 30, 40], scope_name="conv3_top", padding="VALID")  # (b, 44, 44, 40)
      drop3_top = self.dropout(conv3_top, keep_prob=self.keep_prob, scope_name="drop3_top")
      conv4_top = self.conv2d_relu(drop3_top, filter_shape=[3, 3, 40, 40], scope_name="conv4_top", padding="VALID")  # (b, 42, 42, 40)
      drop4_top = self.dropout(conv4_top, keep_prob=self.keep_prob, scope_name="drop4_top")
      conv5_top = self.conv2d_relu(drop4_top, filter_shape=[3, 3, 40, 50], scope_name="conv5_top", padding="VALID")  # (b, 40, 40, 50)
      drop5_top = self.dropout(conv5_top, keep_prob=self.keep_prob, scope_name="drop5_top")
      conv6_top = self.conv2d_relu(drop5_top, filter_shape=[3, 3, 50, 50], scope_name="conv6_top", padding="VALID")  # (b, 38, 38, 50)
      drop6_top = self.dropout(conv6_top, keep_prob=self.keep_prob, scope_name="drop6_top")
      conv7_top = self.conv2d_relu(drop6_top, filter_shape=[3, 3, 50, 60], scope_name="conv7_top", padding="VALID")  # (b, 36, 36, 60)
      drop7_top = self.dropout(conv7_top, keep_prob=self.keep_prob, scope_name="drop7_top")


      # ----- Bottom path ------

      # Crop the input, aggresively pool, then pad to a convenient size
      cropped_input = tf.slice(input, [0, 18, 0, 0], [-1, 196, -1, -1]) # (b, 196, 196, 1)
      blue_box = self.maxpool2d(cropped_input, scope_name='blue_pool') # (b, 98, 98, 1)
      pool1_lower = self.maxpool2d(blue_box, scope_name='pool1_lower') # (b, 49, 49, 1)
      cropped_input2 = tf.slice(pool1_lower, [0, 1, 1, 0], [-1, -1, -1, -1]) # (b, 48, 48, 1)
      pool2_lower = self.maxpool2d(cropped_input2, scope_name='pool2_lower') # (b, 24, 24 1)

      # Convolutions
      conv1_lower = self.conv2d_relu(pool2_lower, filter_shape=[3, 3, 1, 40], scope_name="conv1", padding="VALID")  # (b, 22, 22, 40)
      drop1_lower = self.dropout(conv1_lower, keep_prob=self.keep_prob, scope_name="drop1")
      conv2_lower = self.conv2d_relu(drop1_lower, filter_shape=[3, 3, 40, 50], scope_name="conv2", padding="VALID")  # (b, 20, 20, 50)
      drop2_lower = self.dropout(conv2_lower, keep_prob=self.keep_prob, scope_name="drop2")
      conv3_lower = self.conv2d_relu(drop2_lower, filter_shape=[3, 3, 50, 60], scope_name="conv3", padding="VALID")  # (b, 18, 18, 60)
      drop3_lower = self.dropout(conv3_lower, keep_prob=self.keep_prob, scope_name="drop3")

      # Convolve with same padding
      conv4_lower = self.conv2d_relu(drop3_lower, filter_shape=[3, 3, 60, 60], scope_name="conv4", padding="SAME")  # (b, 18, 18, 60)
      drop4_lower = self.dropout(conv4_lower, keep_prob=self.keep_prob, scope_name="drop4")
      conv5_lower = self.conv2d_relu(drop4_lower, filter_shape=[3, 3, 60, 60], scope_name="conv5", padding="SAME")  # (b, 18, 18, 60)
      drop5_lower = self.dropout(conv5_lower, keep_prob=self.keep_prob, scope_name="drop5")

      # Upsample to correct size for concatenation
      up1 = self.upsample(drop5_lower, scope_name="up1", factor=[2, 2])  # (b, 36, 36, 256)
      deconv1 = self.deconv2d(up1, filter_shape=[2, 2], num_outputs=60, scope_name="deconv1")  # (b, 36, 36, 256)


      # ----- Combine Paths ------
      concat1 = tf.concat([drop7_top, deconv1], axis=3)
      conv_1D_1 = self.conv2d(concat1, filter_shape=[1, 1, 120, 120], scope_name="conv1D_1")  # (b, 36, 36, 128)
      conv_1D_2 = self.conv2d(conv_1D_1, filter_shape=[1, 1, 120, 120], scope_name="conv1D_2")  # (b, 36, 36, 1)
      conv_1D_3 = self.conv2d(conv_1D_2, filter_shape=[1, 1, 120, 1], scope_name="conv1D_3")  # (b, 36, 36, 1)

      # Pad the final output so that it is the same shape as the input
      pad2 = tf.pad(conv_1D_3, tf.constant([[0,0], [100, 96], [100, 60], [0,0]])) # (b, 232, 196, 1)
      out = tf.identity(pad2, name="out")
      
    return out

class DualNet50(NeuralNetwork):
  def __init__(self, input_shape, keep_prob, output_shape, scope_name="DNTest"):
    self.input_shape = input_shape
    self.keep_prob = keep_prob
    self.output_shape = output_shape
    self.scope_name = scope_name

  def build_graph(self, input):


    '''
      - Implement the dual pathway architecture with green_box centered ~100x100, blue_box entire image
      - Spacial dims when you concatenate should be ~50x50 or maybe ~25x25
      - Deconv only on blue_path to get to correct dimensions before concatenating
      - End with 1x1 convolutions for fulling connected layer
      - Run on a couple images with lesions within the green_box
    '''
    with tf.variable_scope(self.scope_name):


      # ----- Top path -----

      # Take a crop out of the middle of the input
      green_box = tf.slice(input, [0, 83, 43, 0], [-1, 50, 50, -1], name="crop1") # (b, 50, 50, 1)

      # Convolutions
      conv1_top = self.conv2d_relu(green_box, filter_shape=[3, 3, 1, 30], scope_name="conv1_top" )  # (b, 50, 50, 30)
      drop1_top = self.dropout(conv1_top, keep_prob=self.keep_prob, scope_name="drop1_top")
      conv2_top = self.conv2d_relu(drop1_top, filter_shape=[3, 3, 30, 30], scope_name="conv2_top")  # (b, 50, 50, 30)
      drop2_top = self.dropout(conv2_top, keep_prob=self.keep_prob, scope_name="drop2_top")
      conv3_top = self.conv2d_relu(drop2_top, filter_shape=[3, 3, 30, 40], scope_name="conv3_top")  # (b, 50, 50, 40)
      drop3_top = self.dropout(conv3_top, keep_prob=self.keep_prob, scope_name="drop3_top")
      conv4_top = self.conv2d_relu(drop3_top, filter_shape=[3, 3, 40, 50], scope_name="conv4_top")  # (b, 50, 50, 40)
      drop4_top = self.dropout(conv4_top, keep_prob=self.keep_prob, scope_name="drop4_top")
      conv5_top = self.conv2d_relu(drop4_top, filter_shape=[3, 3, 50, 50], scope_name="conv5_top")  # (b, 50, 50, 50)
      drop5_top = self.dropout(conv5_top, keep_prob=self.keep_prob, scope_name="drop5_top")
      conv6_top = self.conv2d_relu(drop5_top, filter_shape=[3, 3, 50, 60], scope_name="conv6_top")  # (b, 50, 50, 50)
      drop6_top = self.dropout(conv6_top, keep_prob=self.keep_prob, scope_name="drop6_top")
      conv7_top = self.conv2d_relu(drop6_top, filter_shape=[3, 3, 60, 60], scope_name="conv7_top")  # (b, 50, 50, 60)
      drop7_top = self.dropout(conv7_top, keep_prob=self.keep_prob, scope_name="drop7_top")


      # ----- Bottom path ------

      # Crop the input, aggresively pool, then pad to a convenient size
      cropped_input = tf.slice(input, [0, 18, 0, 0], [-1, 196, -1, -1]) # (b, 196, 196, 1)
      blue_box = self.maxpool2d(cropped_input, scope_name='blue_pool') # (b, 98, 98, 1)
      pool1_lower = self.maxpool2d(blue_box, scope_name='pool1_lower') # (b, 49, 49, 1)
      cropped_input2 = tf.slice(pool1_lower, [0, 1, 1, 0], [-1, -1, -1, -1]) # (b, 48, 48, 1)
      pool2_lower = self.maxpool2d(cropped_input2, scope_name='pool2_lower') # (b, 24, 24 1)


      # Convolutions
      conv1_lower = self.conv2d_relu(pool2_lower, filter_shape=[3, 3, 1, 40], scope_name="conv1")  # (b, 24, 24, 40)
      drop1_lower = self.dropout(conv1_lower, keep_prob=self.keep_prob, scope_name="drop1")
      conv2_lower = self.conv2d_relu(drop1_lower, filter_shape=[3, 3, 40, 40], scope_name="conv2")  # (b, 24, 24, 40)
      drop2_lower = self.dropout(conv2_lower, keep_prob=self.keep_prob, scope_name="drop2")
      conv3_lower = self.conv2d_relu(drop2_lower, filter_shape=[3, 3, 40, 50], scope_name="conv3")  # (b, 24, 24, 50)
      drop3_lower = self.dropout(conv3_lower, keep_prob=self.keep_prob, scope_name="drop3")
      conv4_lower = self.conv2d_relu(drop3_lower, filter_shape=[3, 3, 50, 50], scope_name="conv4")  # (b, 24, 24, 60)
      drop4_lower = self.dropout(conv4_lower, keep_prob=self.keep_prob, scope_name="drop4")
      conv5_lower = self.conv2d_relu(drop4_lower, filter_shape=[3, 3, 50, 60], scope_name="conv5")  # (b, 24, 24, 60)
      drop5_lower = self.dropout(conv5_lower, keep_prob=self.keep_prob, scope_name="drop5")

      # Upsample to the correct size for concatenation
      up1 = self.upsample3(drop5_lower, scope_name="up1", factor=[2, 2])  # (b, 50, 50, 256)
      deconv1 = self.deconv2d(up1, filter_shape=[2, 2], num_outputs=60, scope_name="deconv1")  # (b, 50, 50, 60)


      # ----- Combine Paths ------
      concat1 = tf.concat([drop7_top, deconv1], axis=3)
      conv_1D_1 = self.conv2d(concat1, filter_shape=[1, 1, 120, 120], scope_name="conv1D_1")  # (b, 50, 50, 120)
      drop_fc1 = self.dropout(conv_1D_1, keep_prob=self.keep_prob, scope_name="drop_fc1") 
      conv_1D_2 = self.conv2d(drop_fc1, filter_shape=[1, 1, 120, 120], scope_name="conv1D_2")  # (b, 50, 50, 120)
      drop_fc2 = self.dropout(conv_1D_2, keep_prob=self.keep_prob, scope_name="drop_fc2")
      conv_1D_4 = self.conv2d(drop_fc2, filter_shape=[1, 1, 120, 1], scope_name="conv1D_4")  # (b, 50, 50, 1)

      # Pad the final output so that it is the same shape as the input
      pad2 = tf.pad(conv_1D_4, tf.constant([[0,0], [83, 99], [43, 103], [0,0]])) # (b, 232, 196, 1)
      out = tf.identity(pad2, name="out")

      print(out.shape)

    return out

  def upsample3(self, input, scope_name, factor=[2, 2]):
    size = [int(input.shape[1] * factor[0]) + 2, int(input.shape[2] * factor[1]) + 2]
    with tf.variable_scope(scope_name):
      out = tf.image.resize_bilinear(input, size=size, align_corners=None, name="out")
    return out


class DualNetMultiWindow50(NeuralNetwork):
  def __init__(self, input_shape, keep_prob, output_shape, scope_name="unet"):
    self.input_shape = input_shape
    self.keep_prob = keep_prob
    self.output_shape = output_shape
    self.scope_name = scope_name

  def build_graph(self, input):
    '''
      - Implement the dual pathway architecture with green_box centered ~100x100, blue_box entire image
      - Spacial dims when you concatenate should be ~50x50 or maybe ~25x25
      - Deconv only on blue_path to get to correct dimensions before concatenating
      - End with 1x1 convolutions for fulling connected layer
      - Run on a couple images with lesions within the green_box
    '''

    with tf.variable_scope(self.scope_name, reuse=tf.AUTO_REUSE):

      padded_input = tf.pad(input, tf.constant([[0,0], [9,9], [2,2], [0,0]])) #(b, 250, 200, 1)

      # A list of values to use to offset the green_box from the top left of the image,
      # given as a list of (vertical, horizonal) pairs
      offsets = [(25, 25), (25, 75), (25, 125), (75, 25), (75, 75), (75, 125), (125, 25), (125, 75), (125, 125), (175, 25), (175, 75), (175, 125)]
      offset_outputs = []

      #print('offset coordinates are', offsets[0] + (116,98), offsets[1] + (116, 98))

      for offset in offsets:

        # ----- Top path -----
        # Take a crop out of the input, as specified by the offset

        green_box = tf.slice(padded_input, [0, offset[0], offset[1], 0], [-1, 50, 50, -1], name="crop1") # (b, 50, 50, 1)

        # Initial convolutions
        conv1_top = self.conv2d_relu(green_box, filter_shape=[3, 3, 1, 60], scope_name="conv1_top")  # (b, 50, 50, 60)
        conv2_top = self.conv2d_relu(conv1_top, filter_shape=[3, 3, 60, 60], scope_name="conv2_top")  # (b, 50, 50, 60)
        conv3_top = self.conv2d_relu(conv2_top, filter_shape=[3, 3, 60, 80], scope_name="conv3_top")  # (b, 50, 50, 80)
        conv4_top = self.conv2d_relu(conv3_top, filter_shape=[3, 3, 80, 80], scope_name="conv4_top")  # (b, 50, 50, 80)
        conv5_top = self.conv2d_relu(conv4_top, filter_shape=[3, 3, 80, 100], scope_name="conv5_top")  # (b, 50, 50, 100)
        conv6_top = self.conv2d_relu(conv5_top, filter_shape=[3, 3, 100, 100], scope_name="conv6_top")  # (b, 50, 50, 100)
        conv7_top = self.conv2d_relu(conv6_top, filter_shape=[3, 3, 100, 120], scope_name="conv7_top")  # (b, 50, 50, 120)

        # ----- Bottom path ------
        # Crop the input, aggresively pool, then pad to a convenient size
        cropped_input = tf.slice(padded_input, [0, offset[0]-25, offset[1]-25, 0], [-1, 100, 100, -1]) # (b, 100, 100, 1)
        pool1_lower = self.maxpool2d(cropped_input, scope_name='pool1_lower') # (b, 50, 50, 1)
        conv1_lower = self.conv2d_relu(pool1_lower, filter_shape=[3, 3, 1, 80], scope_name="conv1")  # (b, 25, 25, 80)
        pool1_lower = self.maxpool2d(blue_box, scope_name='pool1_lower') # (b, 25, 25, 80)

        # Convolutions
        conv2_lower = self.conv2d_relu(conv1_lower, filter_shape=[3, 3, 80, 100], scope_name="conv2")  # (b, 25, 25, 100)
        conv3_lower = self.conv2d_relu(conv2_lower, filter_shape=[3, 3, 100, 120], scope_name="conv3")  # (b, 25, 25, 120)
        conv4_lower = self.conv2d_relu(conv3_lower, filter_shape=[3, 3, 120, 120], scope_name="conv4")  # (b, 25, 25, 120)
        conv5_lower = self.conv2d_relu(conv4_lower, filter_shape=[3, 3, 120, 120], scope_name="conv5")  # (b, 25, 25, 120)

        up1 = self.upsample(conv5_lower, scope_name="up1", factor=[2, 2])  # (b, 50, 50, 120)
        deconv1 = self.deconv2d(up1, filter_shape=[2, 2], num_outputs=120, scope_name="deconv1")  # (b, 50, 50, 120)
        deconv2 = self.deconv2d(deconv1, filter_shape=[2, 2], num_outputs=120, scope_name="deconv2")  # (b, 50, 50, 120)

        # ----- Combine Paths ------
        concat1 = tf.concat([conv7_top, deconv1], axis=3)
        conv_1D_1 = self.conv2d(concat1, filter_shape=[1, 1, 240, 240], scope_name="conv1D_1")  # (b, 50, 50, 240)
        conv_1D_2 = self.conv2d(conv_1D_1, filter_shape=[1, 1, 240, 240], scope_name="conv1D_2")  # (b, 50, 50, 240)
        conv_1D_3 = self.conv2d(conv_1D_2, filter_shape=[1, 1, 240, 1], scope_name="conv1D_3")  # (b, 50, 50, 1)

        # Pad the final output so that it is the same shape as the input
        padded_output = tf.pad(conv_1D_3, tf.constant([[0,0], [offset[0] - 8, 190 - offset[0]], [offset[1] - 2, 148 - offset[1]], [0,0]])) # (b, 232, 196, 1)
        offset_outputs.append(padded_output)

      out = tf.identity(tf.reduce_sum(offset_outputs, axis=0), name="out")
      
    return out

class DualNetFC(NeuralNetwork):
  def __init__(self, input_shape, keep_prob, output_shape, scope_name="unet"):
    self.input_shape = input_shape
    self.keep_prob = keep_prob
    self.output_shape = output_shape
    self.scope_name = scope_name

  def build_graph(self, input):
    '''
      - Implement the dual pathway architecture with green_box centered ~100x100, blue_box entire image
      - Spacial dims when you concatenate should be ~50x50 or maybe ~25x25
      - Deconv only on blue_path to get to correct dimensions before concatenating
      - End with 1x1 convolutions for fulling connected layer
      - Run on a couple images with lesions within the green_box
    '''

    with tf.variable_scope(self.scope_name, reuse=tf.AUTO_REUSE):

      padded_input = tf.pad(input, tf.constant([[0,0], [9,9], [2,2], [0,0]])) #(b, 250, 200, 1)

      # A list of values to use to offset the green_box from the top left of the image,
      # given as a list of (vertical, horizonal) pairs
      offsets = [(25, 25), (25, 75), (25, 125), (75, 25), (75, 75), (75, 125), (125, 25), (125, 75), (125, 125), (175, 25), (175, 75), (175, 125)]
      offset_outputs = []

      for offset in offsets:


        # ----- Top path -----
        # Take a crop out of the input, as specified by the offset
        green_box = tf.slice(padded_input, [0, offset[0], offset[1], 0], [-1, 50, 50, -1], name="crop1") # (b, 50, 50, 1)

        # Convolutions
        conv1_top = self.conv2d_relu(green_box, filter_shape=[3, 3, 1, 20], scope_name="conv1_top")  # (b, 50, 50, 20)
        conv2_top = self.conv2d_relu(conv1_top, filter_shape=[3, 3, 20, 20], scope_name="conv2_top")  # (b, 50, 50, 20)
        conv3_top = self.conv2d_relu(conv2_top, filter_shape=[3, 3, 20, 30], scope_name="conv3_top")  # (b, 50, 50, 30)
        conv4_top = self.conv2d_relu(conv3_top, filter_shape=[3, 3, 30, 40], scope_name="conv4_top")  # (b, 50, 50, 40)
        conv5_top = self.conv2d_relu(conv4_top, filter_shape=[3, 3, 40, 50], scope_name="conv5_top")  # (b, 50, 50, 50)
        conv6_top = self.conv2d_relu(conv5_top, filter_shape=[3, 3, 50, 50], scope_name="conv6_top")  # (b, 50, 50, 50)
        conv7_top = self.conv2d_relu(conv6_top, filter_shape=[3, 3, 50, 60], scope_name="conv7_top")  # (b, 50, 50, 60)


        # ----- Bottom path ------
        # Crop the input, aggresively pool, then pad to a convenient size
        cropped_input = tf.slice(padded_input, [0, offset[0]-25, offset[1]-25, 0], [-1, 100, 100, -1]) # (b, 100, 100, 1)
        pool1_lower = self.maxpool2d(cropped_input, scope_name='pool1_lower') # (b, 50, 50, 1)
        conv1_lower = self.conv2d_relu(pool1_lower, filter_shape=[3, 3, 1, 20], scope_name="conv1")  # (b, 25, 25, 20)
        pool2_lower = self.maxpool2d(conv1_lower, scope_name='pool2_lower') # (b, 25, 25, 20)
        conv2_lower = self.conv2d_relu(pool2_lower, filter_shape=[3, 3, 20, 30], scope_name="conv2")  # (b, 25, 25, 30)

        # Fully connected layers
        reshape1 = tf.reshape(conv2_lower, shape=[-1, 25*25*30])  # (b, 18750)
        fc1 = self.fc(reshape1, output_shape=512, scope_name="fc1") # (b, 512)
        drop_fc1 = self.dropout(fc1, keep_prob=self.keep_prob, scope_name="drop_fc1")
        fc2 = self.fc(drop_fc1, output_shape=128, scope_name="fc2") # (b, 128)
        drop_fc2 = self.dropout(fc2, keep_prob=self.keep_prob, scope_name="drop_fc2")
        fc3 = self.fc(drop_fc2, output_shape=512, scope_name="fc3") # (b, 512)
        drop_fc3 = self.dropout(fc3, keep_prob=self.keep_prob, scope_name="drop_fc3")
        fc4 = self.fc(drop_fc3, output_shape=25*25*30, scope_name="fc4") # (b, 1875)
        reshape2 = tf.reshape(fc4, shape=[-1, 25, 25, 30])

        # Upsample to correct size
        up1 = self.upsample(reshape2, scope_name="up1", factor=[2, 2])  # (b, 50, 50, 30)
        deconv1 = self.deconv2d(up1, filter_shape=[2, 2], num_outputs=40, scope_name="deconv1")  # (b, 50, 50, 40)
        deconv2 = self.deconv2d(deconv1, filter_shape=[2, 2], num_outputs=60, scope_name="deconv2")  # (b, 50, 50, 60)

        # ----- Combine Paths ------
        concat1 = tf.concat([conv7_top, deconv2], axis=3)
        conv_1D_1 = self.conv2d(concat1, filter_shape=[1, 1, 120, 60], scope_name="conv1D_1")  # (b, 50, 50, 120)
        conv_1D_2 = self.conv2d(conv_1D_1, filter_shape=[1, 1, 60, 60], scope_name="conv1D_2")  # (b, 50, 50, 60)
        conv_1D_3 = self.conv2d(conv_1D_2, filter_shape=[1, 1, 60, 1], scope_name="conv1D_3")  # (b, 50, 50, 1)

        # Pad the final output so that it is the same shape as the input
        padded_output = tf.pad(conv_1D_3, tf.constant([[0,0], [offset[0] - 8, 190 - offset[0]], [offset[1] - 2, 148 - offset[1]], [0,0]])) # (b, 232, 196, 1)
        offset_outputs.append(padded_output)

      # Pull together outputs from different slices
      out = tf.identity(tf.reduce_sum(offset_outputs, axis=0), name="out")
      
    return out

class OneNetFC(NeuralNetwork):
  def __init__(self, input_shape, keep_prob, output_shape, scope_name="unet"):
    self.input_shape = input_shape
    self.keep_prob = keep_prob
    self.output_shape = output_shape
    self.scope_name = scope_name

  def build_graph(self, input):
    '''
      - Implement the dual pathway architecture with green_box centered ~100x100, blue_box entire image
      - Spacial dims when you concatenate should be ~50x50 or maybe ~25x25
      - Deconv only on blue_path to get to correct dimensions before concatenating
      - End with 1x1 convolutions for fulling connected layer
      - Run on a couple images with lesions within the green_box
    '''

    with tf.variable_scope(self.scope_name, reuse=tf.AUTO_REUSE):

      padded_input = tf.pad(input, tf.constant([[0,0], [9,9], [2,2], [0,0]])) #(b, 250, 200, 1)

      # A list of values to use to offset the green_box from the top left of the image,
      # given as a list of (vertical, horizonal) pairs
      offsets = [(25, 25), (25, 75), (25, 125), (75, 25), (75, 75), (75, 125), (125, 25), (125, 75), (125, 125), (175, 25), (175, 75), (175, 125)]
      offset_outputs = []

      for offset in offsets:

        # ----- Bottom path ------
        # Crop the input, aggresively pool, then pad to a convenient size
        cropped_input = tf.slice(padded_input, [0, offset[0]-25, offset[1]-25, 0], [-1, 100, 100, -1]) # (b, 100, 100, 1)
        pool1_lower = self.maxpool2d(cropped_input, scope_name='pool1_lower') # (b, 50, 50, 1)
        conv1_lower = self.conv2d_relu(pool1_lower, filter_shape=[3, 3, 1, 20], scope_name="conv1")  # (b, 25, 25, 20)
        pool2_lower = self.maxpool2d(conv1_lower, scope_name='pool2_lower') # (b, 25, 25, 20)
        conv2_lower = self.conv2d_relu(pool2_lower, filter_shape=[3, 3, 20, 30], scope_name="conv2")  # (b, 25, 25, 30)

        # Fully connected layers
        reshape1 = tf.reshape(conv2_lower, shape=[-1, 25*25*30])  # (b, 18750)
        fc1 = self.fc(reshape1, output_shape=256, scope_name="fc1") # (b, 256)
        drop_fc1 = self.dropout(fc1, keep_prob=self.keep_prob, scope_name="drop_fc1")
        fc2 = self.fc(drop_fc1, output_shape=128, scope_name="fc2") # (b, 128)
        drop_fc2 = self.dropout(fc2, keep_prob=self.keep_prob, scope_name="drop_fc2")
        fc3 = self.fc(drop_fc2, output_shape=256, scope_name="fc3") # (b, 256)
        drop_fc3 = self.dropout(fc3, keep_prob=self.keep_prob, scope_name="drop_fc3")
        fc4 = self.fc(drop_fc3, output_shape=25*25*30, scope_name="fc4") # (b, 1875)
        reshape2 = tf.reshape(fc4, shape=[-1, 25, 25, 30])

        # Upsample to correct size
        up1 = self.upsample(reshape2, scope_name="up1", factor=[2, 2])  # (b, 50, 50, 30)
        deconv1 = self.deconv2d(up1, filter_shape=[2, 2], num_outputs=40, scope_name="deconv1")  # (b, 50, 50, 40)
        deconv2 = self.deconv2d(deconv1, filter_shape=[2, 2], num_outputs=60, scope_name="deconv2")  # (b, 50, 50, 60)

        conv_1D_1 = self.conv2d(deconv2, filter_shape=[1, 1, 60, 60], scope_name="conv1D_1")  # (b, 50, 50, 60)
        conv_1D_2 = self.conv2d(conv_1D_1, filter_shape=[1, 1, 60, 60], scope_name="conv1D_2")  # (b, 50, 50, 60)
        conv_1D_3 = self.conv2d(conv_1D_2, filter_shape=[1, 1, 60, 1], scope_name="conv1D_3")  # (b, 50, 50, 1)

        # Pad the final output so that it is the same shape as the input
        padded_output = tf.pad(conv_1D_3, tf.constant([[0,0], [offset[0] - 8, 190 - offset[0]], [offset[1] - 2, 148 - offset[1]], [0,0]])) # (b, 232, 196, 1)
        offset_outputs.append(padded_output)

      # Pull together outputs from different slices
      out = tf.identity(tf.reduce_sum(offset_outputs, axis=0), name="out")
      
    return out



class DualNetFC2(NeuralNetwork):
  def __init__(self, input_shape, keep_prob, output_shape, scope_name="unet"):
    self.input_shape = input_shape
    self.keep_prob = keep_prob
    self.output_shape = output_shape
    self.scope_name = scope_name

  def build_graph(self, input):
    '''
      - Implement the dual pathway architecture with green_box centered ~100x100, blue_box entire image
      - Spacial dims when you concatenate should be ~50x50 or maybe ~25x25
      - Deconv only on blue_path to get to correct dimensions before concatenating
      - End with 1x1 convolutions for fulling connected layer
      - Run on a couple images with lesions within the green_box
    '''

    with tf.variable_scope(self.scope_name, reuse=tf.AUTO_REUSE):

      padded_input = tf.pad(input, tf.constant([[0,0], [9,9], [2,2], [0,0]])) #(b, 250, 200, 1)

      # A list of values to use to offset the green_box from the top left of the image,
      # given as a list of (vertical, horizonal) pairs
      offsets = [(25, 25), (25, 75), (25, 125), (75, 25), (75, 75), (75, 125), (125, 25), (125, 75), (125, 125), (175, 25), (175, 75), (175, 125)]
      offset_outputs = []

      # ----- Bottom path ------
      # Crop the input, aggresively pool while padding to a convenient size
      cropped_input = tf.slice(input, [0, 18, 0, 0], [-1, 196, -1, -1]) # (b, 196, 196, 1)
      pool1_lower = self.maxpool2d(cropped_input, scope_name='pool1_lower') # (b, 98, 98, 1)
      conv1_lower = self.conv2d_relu(pool1_lower, filter_shape=[3, 3, 1, 20], scope_name="conv1_lower") # (b, 98, 98, 20)
      pool2_lower = self.maxpool2d(conv1_lower, scope_name='pool2_lower') # (b, 49, 49, 20)
      pool2_lower_pad = tf.pad(pool2_lower, tf.constant([[0,0], [1,0], [1,0], [0,0]])) # (b, 50, 50, 20)
      conv2_lower = self.conv2d_relu(pool2_lower, filter_shape=[3, 3, 20, 30], scope_name="conv2_lower") # (b, 50, 50, 30)
      pool3_lower = self.maxpool2d(conv2_lower, scope_name='pool3_lower') # (b, 25, 25, 30)

      # Fully connected layers
      reshape1 = tf.reshape(pool3_lower, shape=[-1, 25*25*30])  # (b, 18750)
      fc1 = self.fc(reshape1, output_shape=1024, scope_name="fc1_lower") # (b, 1024)     
      fc2 = self.fc(fc1, output_shape=256, scope_name="fc2_lower") # (b, 256)
      fc3 = self.fc(fc2, output_shape=1024, scope_name="fc3_lower") # (b, 1024)   
      fc4 = self.fc(fc3, output_shape=25*25*30, scope_name="fc4_lower") # (b, 25*25*30)
      reshape2 = tf.reshape(fc4, shape=[-1, 25, 25, 30])

      # Upsample to correct size for concatenation with top path
      up1 = self.upsample(reshape2, scope_name="up1", factor=[2, 2])  # (b, 50, 50, 30)
      deconv1 = self.deconv2d(up1, filter_shape=[2, 2], num_outputs=40, scope_name="deconv1")  # (b, 50, 50, 40)
      deconv2 = self.deconv2d(deconv1, filter_shape=[2, 2], num_outputs=60, scope_name="deconv2")  # (b, 50, 50, 60)


      for offset in offsets:

        # ----- Top path -----
        # Take a crop out of the input, as specified by the offset
        green_box = tf.slice(padded_input, [0, offset[0], offset[1], 0], [-1, 50, 50, -1], name="crop1") # (b, 50, 50, 1)

        # Initial convolutions
        conv1_top = self.conv2d_relu(green_box, filter_shape=[3, 3, 1, 20], scope_name="conv1_top")  # (b, 50, 50, 20)
        conv2_top = self.conv2d_relu(conv1_top, filter_shape=[3, 3, 20, 20], scope_name="conv2_top")  # (b, 50, 50, 20)
        conv3_top = self.conv2d_relu(conv2_top, filter_shape=[3, 3, 20, 30], scope_name="conv3_top")  # (b, 50, 50, 30)
        conv4_top = self.conv2d_relu(conv3_top, filter_shape=[3, 3, 30, 40], scope_name="conv4_top")  # (b, 50, 50, 40)
        conv5_top = self.conv2d_relu(conv4_top, filter_shape=[3, 3, 40, 50], scope_name="conv5_top")  # (b, 50, 50, 50)
        conv6_top = self.conv2d_relu(conv5_top, filter_shape=[3, 3, 50, 50], scope_name="conv6_top")  # (b, 50, 50, 50)
        conv7_top = self.conv2d_relu(conv6_top, filter_shape=[3, 3, 50, 60], scope_name="conv7_top")  # (b, 50, 50, 60)


        # ----- Combine Paths ------
        concat1 = tf.concat([conv7_top, deconv2], axis=3)
        conv_1D_1 = self.conv2d(concat1, filter_shape=[1, 1, 120, 60], scope_name="conv1D_1")  # (b, 50, 50, 120)
        conv_1D_2 = self.conv2d(conv_1D_1, filter_shape=[1, 1, 60, 60], scope_name="conv1D_2")  # (b, 50, 50, 120) 
        conv_1D_3 = self.conv2d(conv_1D_2, filter_shape=[1, 1, 60, 1], scope_name="conv1D_3")  # (b, 50, 50, 1)

        # Pad the final output so that it is the same shape as the input
        padded_output = tf.pad(conv_1D_3, tf.constant([[0,0], [offset[0] - 8, 190 - offset[0]], [offset[1] - 2, 148 - offset[1]], [0,0]])) # (b, 232, 196, 1)
        offset_outputs.append(padded_output)

      out = tf.identity(tf.reduce_sum(offset_outputs, axis=0), name="out")
      
    return out

class DualNetVeryFC(NeuralNetwork):
  def __init__(self, input_shape, keep_prob, output_shape, scope_name="unet"):
    self.input_shape = input_shape
    self.keep_prob = keep_prob
    self.output_shape = output_shape
    self.scope_name = scope_name

  def build_graph(self, input):
    '''
      - Implement the dual pathway architecture with green_box centered ~100x100, blue_box entire image
      - Spacial dims when you concatenate should be ~50x50 or maybe ~25x25
      - Deconv only on blue_path to get to correct dimensions before concatenating
      - End with 1x1 convolutions for fulling connected layer
      - Run on a couple images with lesions within the green_box
    '''

    with tf.variable_scope(self.scope_name, reuse=tf.AUTO_REUSE):

      padded_input = tf.pad(input, tf.constant([[0,0], [9,9], [2,2], [0,0]])) #(b, 250, 200, 1)

      # A list of values to use to offset the green_box from the top left of the image,
      # given as a list of (vertical, horizonal) pairs
      offsets = [(25, 25), (25, 75), (25, 125), (75, 25), (75, 75), (75, 125), (125, 25), (125, 75), (125, 125), (175, 25), (175, 75), (175, 125)]
      offset_outputs = []

      #print('offset coordinates are', offsets[0] + (116,98), offsets[1] + (116, 98))

      for offset in offsets:

        # ----- Top path -----
        # Take a crop out of the input, as specified by the offset
        green_box = tf.slice(padded_input, [0, offset[0], offset[1], 0], [-1, 50, 50, -1], name="crop1") # (b, 50, 50, 1)

        # Convolve to introduce more channels, then pass to fully connected layers
        conv1_top = self.conv2d_relu(green_box, filter_shape=[3, 3, 1, 20], scope_name="conv1_top")  # (b, 50, 50, 20)
        reshape1_top = tf.reshape(conv1_top, shape=[-1, 50*50*20])
        fc1_top = self.fc(reshape1_top, output_shape=256, scope_name='fc1_top')
        fc2_top = self.fc(fc1_top, output_shape=128, scope_name='fc2_top')
        fc3_top = self.fc(fc2_top, output_shape=256, scope_name='fc3_top')
        fc4_top = self.fc(fc3_top, output_shape=50*50*20, scope_name='fc4_top')
        reshape2_top = tf.reshape(fc4_top, shape=[-1, 50, 50, 20])
        conv5_top = self.conv2d_relu(reshape2_top, filter_shape=[3, 3, 20, 60], scope_name="conv5_top")  # (b, 50, 50, 60)
       

        # ----- Bottom path ------
        # Crop the input, aggresively pool, then pad to a convenient size
        cropped_input = tf.slice(padded_input, [0, offset[0]-25, offset[1]-25, 0], [-1, 100, 100, -1]) # (b, 100, 100, 1)
        pool1_lower = self.maxpool2d(cropped_input, scope_name='pool1_lower') # (b, 50, 50, 1)
        conv1_lower = self.conv2d_relu(pool1_lower, filter_shape=[3, 3, 1, 20], scope_name="conv1")  # (b, 50, 50, 20)
        pool2_lower = self.maxpool2d(conv1_lower, scope_name='pool2_lower') # (b, 25, 25, 20)
        conv2_lower = self.conv2d_relu(pool2_lower, filter_shape=[3, 3, 20, 30], scope_name="conv2")  # (b, 25, 25, 30)

        reshape1 = tf.reshape(conv2_lower, shape=[-1, 25*25*30])  # (b, 18750)
        fc1 = self.fc(reshape1, output_shape=512, scope_name="fc1") # (b, 1024)
        fc2 = self.fc(fc1, output_shape=256, scope_name="fc2") # (b, 256)
        fc3 = self.fc(fc2, output_shape=512, scope_name="fc3") # (b, 1024)       
        fc4 = self.fc(fc3, output_shape=25*25*30, scope_name="fc4") # (b, 25*25*30)
        reshape2 = tf.reshape(fc4, shape=[-1, 25, 25, 30])

        # Upsample to the correct dimensions for concatenation
        up1 = self.upsample(reshape2, scope_name="up1", factor=[2, 2])  # (b, 50, 50, 30)
        deconv1 = self.deconv2d(up1, filter_shape=[2, 2], num_outputs=60, scope_name="deconv1")  # (b, 50, 50, 60)
       

        # ----- Fully connected layers ------
        concat1 = tf.concat([conv5_top, deconv1], axis=3)
        conv_1D_1 = self.conv2d(concat1, filter_shape=[1, 1, 120, 60], scope_name="conv1D_1")  # (b, 50, 50, 120)
        conv_1D_2 = self.conv2d(conv_1D_1, filter_shape=[1, 1, 60, 60], scope_name="conv1D_2")  # (b, 50, 50, 60)
        conv_1D_3 = self.conv2d(conv_1D_2, filter_shape=[1, 1, 60, 1], scope_name="conv1D_3")  # (b, 50, 50, 1)

        # Pad the final output so that it is the same shape as the input
        padded_output = tf.pad(conv_1D_3, tf.constant([[0,0], [offset[0] - 8, 190 - offset[0]], [offset[1] - 2, 148 - offset[1]], [0,0]])) # (b, 232, 196, 1)
        offset_outputs.append(padded_output)

      out = tf.identity(tf.reduce_sum(offset_outputs, axis=0), name="out")
      
    return out

