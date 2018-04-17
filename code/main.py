import json
import logging
import os
import sys
import tensorflow as tf

from atlas_model import ATLASModel
from split import setup_train_dev_split

# Relative path of the main directory
MAIN_DIR = os.path.relpath(
  os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Relative path of the data directory
DEFAULT_DATA_DIR = os.path.join(MAIN_DIR, "data")

# Relative path of the experiments directory
EXPERIMENTS_DIR = os.path.join(MAIN_DIR, "experiments")

# Training
tf.app.flags.DEFINE_string("experiment_name", "",
                           "Creates a dir with this name in the experiments/ "
                           "directory, to which checkpoints and logs related "
                           "to this experiment will be saved.")
tf.app.flags.DEFINE_integer("gpu", 0,
                            "Sets which GPU to use, if you have multiple.")
tf.app.flags.DEFINE_string("mode", "train",
                           "Options: {train,show_examples}")
tf.app.flags.DEFINE_integer("num_epochs", None,
                            "Sets the number of epochs to train. None means "
                            "train indefinitely.")
tf.app.flags.DEFINE_string("train_dir", "",
                           "Sets the dir to which checkpoints and logs will "
                           "be saved. Defaults to "
                           "experiments/{experiment_name}")

# Split
tf.app.flags.DEFINE_string("cv_type", "lpocv",
                           "Sets the type of cross validation. Options: "
                           "{lpocv,loocv}")
tf.app.flags.DEFINE_integer("p", None,
                           "Sets p for leave-p-out cross-validation. Defaults "
                           "to floor(0.3 * n) where n represents the number "
                           "groups implied by {split_type}; e.g. n=220 for "
                           "by_patient, n=229 for by_scan, n=9 for by_site.")
tf.app.flags.DEFINE_string("split_type", "by_slice",
                           "Sets the type of split between the train and dev "
                           "sets. Options: "
                           "{by_patient,by_scan,by_slice,by_site}")

# Logging
tf.app.flags.DEFINE_integer("eval_every", 500,
                            "How many iterations to do per calculating the "
                            "dice coefficient on the dev set. This operation "
                            "is time-consuming, so should not be done often.")
tf.app.flags.DEFINE_integer("print_every", 1,
                            "How many iterations to do per print.")
tf.app.flags.DEFINE_integer("save_every", 500,
                            "How many iterations to do per save.")
tf.app.flags.DEFINE_integer("keep", None,
                            "How many checkpoints to keep. None means keep "
                            "all. These files are storage-consuming so should "
                            "not be kept in aggregate.")

# Data
tf.app.flags.DEFINE_string("data_dir", DEFAULT_DATA_DIR,
                           "Sets the dir in which to find data for training. "
                           "Defaults to data/.")

# Model
tf.app.flags.DEFINE_integer("image_height", 233, "Sets the image height.")
tf.app.flags.DEFINE_integer("image_width", 197, "Sets the image width.")

# ResNet
tf.app.flags.DEFINE_integer("resnet_size", 34, "Sets the ResNet size.")

# Hyperparameters
tf.app.flags.DEFINE_integer("batch_size", 100, "Sets the batch size")
tf.app.flags.DEFINE_float("dropout", 0.15,
                          "Sets the fraction of units randomly dropped on "
                          "non-recurrent connections.")
tf.app.flags.DEFINE_float("learning_rate", 0.001, "Sets the learning rate.")
tf.app.flags.DEFINE_float("max_gradient_norm", 5.0,
                          "Clips the gradients to this norm.")

FLAGS = tf.app.flags.FLAGS
os.environ["CUDA_VISIBLE_DEVICES"] = str(FLAGS.gpu)


def initialize_model(sess, model, train_dir, expect_exists):
    """
    Initializes model from {train_dir}.

    Inputs:
    - sess: A TensorFlow Session object.
    - model: An ATLASModel object.
    - train_dir: A Python str that represents the train dir.
    - expect_exists: If True, throw an error if no checkpoint is found;
      otherwise, initialize fresh model if no checkpoint is found.
    """
    ckpt = tf.train.get_checkpoint_state(train_dir)
    v2_path = ckpt.model_checkpoint_path + ".index" if ckpt else ""
    if (ckpt and (tf.gfile.Exists(ckpt.model_checkpoint_path)
        or tf.gfile.Exists(v2_path))):
      print(f"Reading model parameters from {ckpt.model_checkpoint_path}")
      model.saver.restore(sess, ckpt.model_checkpoint_path)
    else:
      if expect_exists:
        raise Exception(f"There is no saved checkpoint at {train_dir}")
      else:
        print(f"There is no saved checkpoint at {train_dir}. Creating model "
              f"with fresh parameters.")
        sess.run(tf.global_variables_initializer())


def main(_):
  #############################################################################
  # Configuration                                                             #
  #############################################################################
  # Checks for Python 3.6
  if sys.version_info[0] != 3:
    raise Exception(f"ERROR: You must use Python 3.6 but you are running "
                    f"Python {sys.version_info[0]}")

  # Prints Tensorflow version
  print(f"This code was developed and tested on TensorFlow 1.7.0. "
        f"Your TensorFlow version: {tf.__version__}.")

  if not FLAGS.experiment_name:
    raise Exception("You need to specify an --experiment_name or --train_dir.")
  FLAGS.train_dir = (FLAGS.train_dir
                     or os.path.join(EXPERIMENTS_DIR, FLAGS.experiment_name))

  # Sets GPU settings
  config = tf.ConfigProto()
  config.gpu_options.allow_growth = True

  # Sets logging configuration
  logging.basicConfig(filename=os.path.join(FLAGS.train_dir, "log.txt"),
                      level=logging.INFO)

  #############################################################################
  # Train/dev split and model definition                                      #
  #############################################################################
  # Initializes model
  atlas_model = ATLASModel(FLAGS)

  if FLAGS.mode == "train":
    if not os.path.exists(FLAGS.train_dir):
      os.makedirs(FLAGS.train_dir)

    # Saves a record of flags as a .json file in train_dir
    with open(os.path.join(FLAGS.train_dir, "flags.json"), "w") as fout:
      flags = {k: v.serialize() for k, v in FLAGS.__flags.items()}
      json.dump(flags, fout)

    with tf.Session(config=config) as sess:
      # Loads the most recent model
      initialize_model(sess, atlas_model, FLAGS.train_dir, expect_exists=False)

      # Trains the model
      atlas_model.train(sess, *setup_train_dev_split(FLAGS))


if __name__ == "__main__":
  tf.app.run()
