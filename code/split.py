import glob
import json
import logging
import math
import os
import random
import re


def _add_paths_to_lists(input_paths_list,
                        target_mask_paths_list,
                        prefix,
                        merge_target_masks):
  """
  Retrieves the 1+ target mask paths corresponding to each input path in
  {input_paths_list} and appends the the target mask path(s) to
  {target_mask_path_list}.

  For example, suppose:
  - input_paths_list = [
      [
        "../data/ATLAS_R1.1/Site1/031806/t01/031806_t1w_deface_stx/image-slice001.jpg",
        "../data/ATLAS_R1.1/Site1/031806/t01/031806_t1w_deface_stx/image-slice002.jpg",
        "../data/ATLAS_R1.1/Site1/031806/t01/031806_t1w_deface_stx/image-slice003.jpg",
        ...
      ],
      ...
    ]

  Patient 031806 scan t01 has two lesion masks; if {merge_target_masks} is
  True, this function results in:
  - target_mask_paths_list = [
      [
        [
          "../data/ATLAS_R1.1/Site1/031806/t01/031806_LesionSmooth_stx/image-slice001.jpg",
          "../data/ATLAS_R1.1/Site1/031806/t01/031806_LesionSmooth_1_stx/image-slice001.jpg"
        ],
        [
          "../data/ATLAS_R1.1/Site1/031806/t01/031806_LesionSmooth_stx/image-slice002.jpg",
          "../data/ATLAS_R1.1/Site1/031806/t01/031806_LesionSmooth_1_stx/image-slice002.jpg"
        ],
        [
          "../data/ATLAS_R1.1/Site1/031806/t01/031806_LesionSmooth_stx/image-slice003.jpg",
          "../data/ATLAS_R1.1/Site1/031806/t01/031806_LesionSmooth_1_stx/image-slice003.jpg"
        ],
        ...
      ],
      ...
    ]
  """
  for input_paths in input_paths_list:
    target_mask_paths = []
    # {input_paths} is a list of input paths with length the depth of the
    # volume for {FLAGS.use_volumetric} as True or length 1 otherwise
    for input_path in input_paths:
      input_path_regex = ("Site(?P<site_id>[0-9]+)(?:\\\|/)+"
                          "(?P<patient_id>[0-9]+)(?:\\\|/)+"
                          "(?P<scan_id>t[0-9]+)(?:\\\|/)+"
                          "[0-9]+_t1w_deface_stx(?:\\\|/)+"
                          "image-slice(?P<slice_id>[0-9]+).jpg")
      site_id, patient_id, scan_id, slice_id = re.findall(input_path_regex,
                                                          input_path)[0]
      target_mask_path_regex = (f"Site{site_id}/{patient_id}/{scan_id}/"
                                f"{patient_id}_LesionSmooth_*/"
                                f"image-slice{slice_id}.jpg")
      target_mask_paths_for_slice = glob.glob(os.path.join(prefix, target_mask_path_regex),
                                              recursive=True)
      if merge_target_masks:
        target_mask_paths.append(target_mask_paths_for_slice)
      else:
        # Adds separate paths for each target mask
        raise NotImplementedError()
    target_mask_paths_list.append(target_mask_paths)

def setup_train_dev_split(FLAGS):
  """
  Splits the dataset into training and development sets. If
  {os.path.join(FLAGS.train_dir, "split.json")} exists, then loads the splits
  from the file. Otherwise, performs the split and writes it to file.

  Inputs:
  - FLAGS: A _FlagValuesWrapper object.

  Outputs:
  - A Python tuple of {train_input_paths}, {train_target_mask_paths},
    {dev_input_paths}, and {dev_target_mask_paths}. The input paths
    consist of Python lists of lists. Each inner list represents a single
    example. If {FLAGS.use_volumetric} is True, then this list contains
    multiple paths that together represent a volume of slices.

    e.g. train_input_paths = [
      [
        "../data/ATLAS_R1.1/Site1/031806/t01/031806_t1w_deface_stx/image-slice001.jpg",
        "../data/ATLAS_R1.1/Site1/031806/t01/031806_t1w_deface_stx/image-slice002.jpg",
        "../data/ATLAS_R1.1/Site1/031806/t01/031806_t1w_deface_stx/image-slice003.jpg",
        ...
      ],
      ...
    ]

    Otherwise, if {FLAGS.use_volumetric} is False, then this inner list
    contains a single path that represents a single slice.

    e.g. train_input_paths = [
      ["../data/ATLAS_R1.1/Site1/031806/t01/031806_t1w_deface_stx/image-slice001.jpg"],
      ["../data/ATLAS_R1.1/Site1/031806/t01/031806_t1w_deface_stx/image-slice002.jpg"],
      ...
    ]

    The target mask paths consist of Python lists of lists of lists. Each
    inner-most list represents the target mask paths corresponding to the
    input path (more than 1 if multiple lesions in the slice). The number of
    inner-most lists depends on the number of slices per example. If
    {FLAGS.use_volumetric} is True,

    e.g. train_target_mask_paths = [
      [
        [
          "../data/ATLAS_R1.1/Site1/031806/t01/031806_LesionSmooth_stx/image-slice001.jpg",
          "../data/ATLAS_R1.1/Site1/031806/t01/031806_LesionSmooth_1_stx/image-slice001.jpg"
        ],
        [
          "../data/ATLAS_R1.1/Site1/031806/t01/031806_LesionSmooth_stx/image-slice002.jpg",
          "../data/ATLAS_R1.1/Site1/031806/t01/031806_LesionSmooth_1_stx/image-slice002.jpg"
        ],
        [
          "../data/ATLAS_R1.1/Site1/031806/t01/031806_LesionSmooth_stx/image-slice003.jpg",
          "../data/ATLAS_R1.1/Site1/031806/t01/031806_LesionSmooth_1_stx/image-slice003.jpg"
        ],
        ...
      ],
      ...
    ]

    Otherwise, if {FLAGS.use_volumetric} is False,

    e.g. train_target_mask_paths = [
      [
        [
          "../data/ATLAS_R1.1/Site1/031806/t01/031806_LesionSmooth_stx/image-slice001.jpg",
          "../data/ATLAS_R1.1/Site1/031806/t01/031806_LesionSmooth_1_stx/image-slice001.jpg"
        ]
      ],
      [
        [
          "../data/ATLAS_R1.1/Site1/031806/t01/031806_LesionSmooth_stx/image-slice002.jpg",
          "../data/ATLAS_R1.1/Site1/031806/t01/031806_LesionSmooth_1_stx/image-slice002.jpg"
        ]
      ],
      [
        [
          "../data/ATLAS_R1.1/Site1/031806/t01/031806_LesionSmooth_stx/image-slice003.jpg",
          "../data/ATLAS_R1.1/Site1/031806/t01/031806_LesionSmooth_1_stx/image-slice003.jpg"
        ]
      ],
      ...
    ]
  """
  # Saves a record of flags as a .json file in {train_dir}
  split_filename = os.path.join(FLAGS.train_dir, "split.json")
  if os.path.exists(split_filename):
    logging.info(f"Reading split from {split_filename}...")
    with open(split_filename, "r") as fin:
      split = json.load(fin)
    return (
      split["train_input_paths"],
      split["train_target_mask_paths"],
      split["dev_input_paths"],
      split["dev_target_mask_paths"]
    )

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
    assert(FLAGS.input_regex == None)
    # If {FLAGS.use_volumetric}, then {train_input_paths} and {dev_input_paths}
    # look like:
    #   [
    #     [
    #       "../data/ATLAS_R1.1/Site1/031806/t01/031806_t1w_deface_stx/image-slice001.jpg",
    #       "../data/ATLAS_R1.1/Site1/031806/t01/031806_t1w_deface_stx/image-slice002.jpg",
    #       "../data/ATLAS_R1.1/Site1/031806/t01/031806_t1w_deface_stx/image-slice003.jpg",
    #       ...
    #     ],
    #     ...
    #   ]
    # Otherwise, they look like:
    #   [
    #     ["../data/ATLAS_R1.1/Site1/031806/t01/031806_t1w_deface_stx/image-slice001.jpg"],
    #     ["../data/ATLAS_R1.1/Site1/031806/t01/031806_t1w_deface_stx/image-slice002.jpg"],
    #     ...
    #   ]
    input_paths_regex = "Site*/*"
    patient_paths = glob.glob(os.path.join(prefix, input_paths_regex),
                              recursive=True)
    random.shuffle(patient_paths)
    input_paths = train_input_paths
    for idx, patient_path in enumerate(patient_paths):
      if idx >= n - FLAGS.p:
        input_paths = dev_input_paths

      slice_paths_by_patient = sorted(
        glob.glob(f"{patient_path}/**/*_t1w_deface_stx/*.jpg", recursive=True))
      if FLAGS.use_volumetric:
        input_paths.append(slice_paths_by_patient)
      else:
        for slice_path_by_patient in slice_paths_by_patient:
          input_paths.append([slice_path_by_patient])
  elif FLAGS.split_type == "by_scan":
    input_paths_regex = "Site*/*/*"
    scan_paths = glob.glob(os.path.join(prefix, input_paths_regex),
                           recursive=True)
    random.shuffle(scan_paths)
    input_paths = train_input_paths
    for idx, scan_path in enumerate(scan_paths):
      if idx >= n - FLAGS.p:
        input_paths = dev_input_paths

      slice_paths_by_scan = sorted(
        glob.glob(f"{scan_path}/**/*_t1w_deface_stx/*.jpg", recursive=True))
      if FLAGS.use_volumetric:
        input_paths.append(slice_paths_by_scan)
      else:
        for slice_path_by_scan in slice_paths_by_scan:
          input_paths.append([slice_path_by_scan])
  elif FLAGS.split_type == "by_slice":
    # A shuffled list of all paths to input (MRI) slices
    if FLAGS.input_regex == None:
      input_paths_regex = "Site*/**/*_t1w_deface_stx/*.jpg"
    else:
      input_paths_regex = FLAGS.input_regex

    slice_paths = glob.glob(os.path.join(prefix, input_paths_regex),
                            recursive=True)
    random.shuffle(slice_paths)
    input_paths = train_input_paths
    if FLAGS.input_regex == None:
      # Bins the first p into the dev set; bins the others into the training set
      for idx, slice_path in enumerate(slice_paths):
        if idx >= FLAGS.p:
          input_paths = dev_input_paths

        if FLAGS.use_volumetric:
          raise ValueError("Cannot {FLAGS.use_volumetric} when "
                           "{FLAGS.split_type} is by_slice.")
        else:
          input_paths.append([slice_path])
    else:
      for slice_path in slice_paths:
        train_input_paths.append([slice_path])
        dev_input_paths.append([slice_paths])
  elif FLAGS.split_type == "by_site":
    pass  # TODO
  else:
    raise ValueError(f"Specified unknown FLAGS.split_type={FLAGS.split_type}.")

  _add_paths_to_lists(train_input_paths,
                      train_target_mask_paths,
                      prefix,
                      FLAGS.merge_target_masks)
  _add_paths_to_lists(dev_input_paths,
                      dev_target_mask_paths,
                      prefix,
                      FLAGS.merge_target_masks)

  # Saves a record of split as a .json file in {train_dir}
  with open(os.path.join(FLAGS.train_dir, "split.json"), "w") as fout:
    split = {
      "train_input_paths": train_input_paths,
      "train_target_mask_paths": train_target_mask_paths,
      "dev_input_paths": dev_input_paths,
      "dev_target_mask_paths": dev_target_mask_paths
    }
    json.dump(split, fout)

  return (
    train_input_paths,
    train_target_mask_paths,
    dev_input_paths,
    dev_target_mask_paths,
  )

