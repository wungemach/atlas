#DualNetFC Network
## Introduction
This is a joint-work from Weston Wungemach (wungemach 'at' stanford.edu) and Beite Zhu (jupiterz 'at' stanford.edu) for the CS230 2018 Spring final project. Our work focuses on a segmentation model to detect the brain lesions from the ATLAS data set.  This repository starts as fork from the repo setup by our mentor David Eng's branch (https://github.com/gnedivad/atlas). Details for the setup are presented at the next section fo the README.

## Model Architecture


# ATLAS (Instructions from David Eng's master branch)
## Setup
Visit the [ATLAS website](http://fcon_1000.projects.nitrc.org/indi/retro/atlas.html) and complete the [form](https://docs.google.com/forms/d/e/1FAIpQLSclH8padHr9zwdQVx9YY_yeM_4OqD1OQFvYcYpAQKaqC6Vscg/viewform) to request access for the dataset.

I've arranged for the maintainers of the dataset to approve all requests from students in our course, so include your affiliation with CS 230 in the Description field. You will receive an email from the Neural Plasticity and Neurorehabilitation Laboratory with the encryption key. I received mine within one business day.

### Mac OS or Linux users
Clone the repo then run the `get-started.sh` script. You will be prompted to enter the encryption key emailed to you.
```bash
$ git clone --recurse-submodules https://github.com/gnedivad/atlas (#To clone the master branch)
```
To clone ours:
```bash
$ git clone --recurse-submodules https://github.com/wungemach/atlas
$ cd atlas
$ sh get-started.sh
```

### Windows users
Create a virtual environment and install the requirements.
```bash
> conda create -n py36 python=3.6 anaconda
> activate py36
(py36) > cd atlas
(py36) > pip install -r requirements.txt
```

Download [OpenSSL](https://slproweb.com/products/Win32OpenSSL.html) and follow the installation instructions. I installed `Win64 OpenSSL v1.0.2o Light` to the location `C:\OpenSSL-Win64`.

Download the [ATLAS encrypted compressed dataset](ftp://www.nitrc.org/fcon_1000/htdocs/indi/retro/ATLAS/releases/R1.1/ATLAS_R1.1_encrypted.tar.gz) and decrypt it. You will be prompted to enter the encryption key emailed to you.
```bash
(py36) > set OPENSSL_CONF=C:\OpenSSL-Win64\bin\openssl.cfg
(py36) > C:\OpenSSL-Win64\openssl aes-256-cbc -d -a -in <encrypted_filename> -out <decrypted_filename>
```

Unpack the decrypted compressed dataset and put it into the `atlas/data/` directory. Please email me for further instructions if you have issues with this step.

## How to run experiments
All functionality starts in `atlas/code/main.py`. Overall, there are two modes:
- `train`: Trains a model (default).
- `eval`: Evaluates a model.

### Your first experiment
Running the following command will start an experiment called `0001` training the model `ZeroATLASModel` for 1 epoch. Recall that an epoch completes when the model has been trained on the entire training set. There might exist many steps per epoch, depending on the batch size. For example, if the training set has 30,000 examples and the batch size is set to 100, then an epoch requires 300 steps to complete.
```bash
(py36) $ cd atlas/code
(py36) $ python main.py --experiment_name=0001 --model_name=ZeroATLASModel --num_epochs=1 --use_fake_target_masks
```

This particular `ZeroATLASModel` predicts 0 for the entire mask no matter what (i.e. it never predicts the presence of a lesion), which performs well when using fake target masks, which manually sets all masks to 0. You can see its stellar performance in `experiments/0001/log.txt`, where a loss close to `0.0` should be printed every training iteration. It's helpful to use the `--use_fake_target_masks` test as a sanity check for new models that you implement.

### Your first real experiment
Running the following command will start an experiment called `0002` training the default encoder-decoder `ATLASModel` for 10 epochs. It will evaluate the performance of the model on the development set every 100 steps, print logs every 1 step, save the model every 100 steps, and written summaries to TensorBoard every 10 steps.
```bash
(py36) $ python main.py --experiment_name=0002 --num_epochs=10 --eval_every=100 --print_every=1 --save_every=100 --summary_every=10
```
You can track the progress of training one of two ways:
- `log.txt`: shows log information written about loss written during training.
- TensorBoard: displays graphical information about loss scalars and images written during training. For example, the TensorBoard images tab shows triplets of MRI slices, target lesion masks, and predicted lesion masks (from left to right).
  ![TensorBoard Triplets](/images/tensorboard_triplets.png)

If you want to stop training and resume it at a later time, quit out of the command. Running the following command restores the weights from the latest checkpoint and reads the dataset split used in the previous training session from `experiments/0002/split.json` (resets the global step but does not restore the values of the flags from `experiments/0002/flags.json`).
```bash
(py36) $ python main.py --experiment_name=0002
```

To use the same flags as the previous training session, specify them explicitly as follows:
```bash
(py36) $ python main.py --experiment_name=0002 --num_epochs=10 --eval_every=100 --print_every=1 --save_every=100 --summary_every=10
```

Checkpoints will be saved every `save_every` iterations, as well as the final iteration of each epoch.

## Helpful flags for sanity checking
- `--use_fake_target_masks`: sets all target masks to entirely zeros, a label that all models should be able to learn. Here, I verify that model `ZeroATLASModel`, which predicts 0 for all masks, achieves a low loss.
  ```bash
  $ python main.py --experiment_name=zero --model_name=ZeroATLASModel --use_fake_target_masks
  ```

- `--input_regex`: sets the regex to use for input paths. If set, `FLAGS.p` will be ignored and train and dev sets will use this same input regex. For small enough sets, all models should be able to overfit the target masks. Here, I verify that the baseline model can overfit to separate lesion masks for a single slice.
  ```bash
  $ python main.py --experiment_name=regex --input_regex=Site2/031844/t01/031844_t1w_deface_stx/image-slice102.jpg
  ```

  It might help to use this flag to experiment with the ability of a model to overfit the target masks as a function of the number of examples. Here, I verfiy that the baseline model can overfit to separate lesion masks for a several patients at the same site.
  ```bash
  $ python main.py --experiment_name=regex --input_regex=Site2/03184*/t01/03184*_t1w_deface_stx/*.jpg
  ```

## Helpful flags to toggle
- `--merge_target_masks`: merges target masks (if more than one exists) for each slice into a single mask. Here, I verify that the baseline model can overfit to a merged lesion mask for a single slice.
  ```bash
  $ python main.py --experiment_name=merge --input_regex=Site2/031844/t01/031844_t1w_deface_stx/image-slice102.jpg --merge_target_masks
  ```

- `--dev_num_samples`: sets the number of examples to evaluate from the dev set (and accelerate training).
  ```bash
  $ python main.py --experiment_name=quick_dev --dev_num_samples=1000
  ```

# Miscellaneous information about the dataset
- 76734 lesion mask slices.
