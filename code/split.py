import glob
import math
import os
import random
import re


def _add_paths_to_lists(input_paths_by_slice,
                        input_paths_list,
                        target_mask_paths_list,
                        prefix):
  """
  Retrieves the 1+ target mask paths corresponding to each input path in
  {input_paths_by_slice} and, for each target mask path, appends the input path
  to {input_paths_list} and the target mask path to {target_mask_path_list}.

  For example, suppose:
  - input_paths_by_slice = [
      "../data/ATLAS_R1.1/Site1/031806/t01/031806_t1w_deface_stx/image-slice101.jpg"
    ]

  Patient 031806 scan t01 has two lesion masks, so this function would append
  paths to the lists, resulting in:
  - input_paths_list = [
      "../data/ATLAS_R1.1/Site1/031806/t01/031806_t1w_deface_stx/image-slice101.jpg",
      "../data/ATLAS_R1.1/Site1/031806/t01/031806_t1w_deface_stx/image-slice101.jpg"
    ]
  - target_mask_paths_list = [
      "../data/ATLAS_R1.1/Site1/031806/t01/031806_LesionSmooth_stx/image-slice101.jpg",
      "../data/ATLAS_R1.1/Site1/031806/t01/031806_LesionSmooth_1_stx/image-slice101.jpg"
    ]
  """
  for input_path in input_paths_by_slice:
    input_path_regex = ("Site(?P<site_id>[0-9]+)/(?P<patient_id>[0-9]+)/"
                        "(?P<scan_id>t[0-9]+)/[0-9]+_t1w_deface_stx/"
                        "image-slice(?P<slice_id>[0-9]+).jpg")
    site_id, patient_id, scan_id, slice_id = re.findall(input_path_regex,
                                                        input_path)[0]
    target_mask_path_regex = (f"Site{site_id}/{patient_id}/{scan_id}/"
                              f"{patient_id}_LesionSmooth_*/"
                              f"image-slice{slice_id}.jpg")
    target_mask_paths = glob.glob(os.path.join(prefix, target_mask_path_regex),
                                  recursive=True)
    for target_mask_path in target_mask_paths:
      input_paths_list.append(input_path)
      target_mask_paths_list.append(target_mask_path)

def setup_train_dev_split(FLAGS):
  if FLAGS.split_type == "by_patient":
    n = 220
  elif FLAGS.split_type == "by_scan":
    n = 229
  elif FLAGS.split_type == "by_slice":
    # find . -type f -wholename "*Site*/*/*/*_t1w_deface_stx/*.jpg" | wc -l
    n = 43281
  elif FLAGS.split_type == "by_site":
    n = 9
  else:
    raise ValueError(f"Specified unknown FLAGS.split_type={FLAGS.split_type}.")

  if FLAGS.cv_type == "loocv":
    FLAGS.p = 1
  elif FLAGS.p == None:
    FLAGS.p = math.floor(0.3 * n)
  # else: it's been set manually

  train_input_paths = []
  train_target_mask_paths = []
  dev_input_paths = []
  dev_target_mask_paths = []

  prefix = os.path.join(FLAGS.data_dir, "ATLAS_R1.1")

  # Select p out of n items, specified by FLAGS.split_type
  if FLAGS.split_type == "by_patient":
    pass  # TODO
  elif FLAGS.split_type == "by_scan":
    pass  # TODO
  elif FLAGS.split_type == "by_slice":
    # A shuffled list of all paths to input (MRI) slices
    input_paths_regex = "Site*/**/*_t1w_deface_stx/*.jpg"
    input_paths_by_slice = glob.glob(os.path.join(prefix, input_paths_regex),
                                     recursive=True)
    random.shuffle(input_paths_by_slice)

    # Bins the first p into the dev set; bins the others into the training set
    train_input_paths_by_slice = input_paths_by_slice[FLAGS.p:]
    dev_input_paths_by_slice = input_paths_by_slice[:FLAGS.p]

  elif FLAGS.split_type == "by_site":
    pass  # TODO
  else:
    raise ValueError(f"Specified unknown FLAGS.split_type={FLAGS.split_type}.")

  _add_paths_to_lists(train_input_paths_by_slice,
                      train_input_paths,
                      train_target_mask_paths,
                      prefix)
  _add_paths_to_lists(dev_input_paths_by_slice,
                      dev_input_paths,
                      dev_target_mask_paths,
                      prefix)

  return (
    train_input_paths,
    train_target_mask_paths,
    dev_input_paths,
    dev_target_mask_paths,
  )

