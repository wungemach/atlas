import glob
import math
import os
import re

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
    # find . -type f -wholename "*Site*/*/*/*_t1w_deface_stx/*.jpg" | wc -l
    input_paths_by_slice = glob.glob(os.path.join(prefix, "Site*/**/*_t1w_deface_stx/*.jpg"), recursive=True)
    random.shuffle(input_paths_by_slice)
    train_input_paths_by_slice = input_paths_by_slice[p:]
    dev_input_paths_by_slice = input_paths_by_slice[:p]

    for input_path in train_input_paths_by_slice:
      site_id, patient_id, scan_id, slice_id = re.findall(
        "Site(?P<site_id>[0-9]+)/(?P<patient_id>[0-9]+)/(?P<scan_id>t[0-9]+)/[0-9]+_t1w_deface_stx/image-slice(?P<slice_id>[0-9]+).jpg",
        input_path)[0]
      target_mask_paths = glob.glob(os.path.join(prefix, f"Site{site_id}/{patient_id}/{scan_id}/{patient_id}_LesionSmooth_*/image-slice{slice_id}.jpg"), recursive=True)
      for target_mask_path in target_mask_paths:
        train_input_paths.append(input_path)
        train_target_mask_paths.append(target_mask_path)

    for input_path in dev_input_paths_by_slice:
      site_id, patient_id, scan_id, slice_id = re.findall(
        "Site(?P<site_id>[0-9]+)/(?P<patient_id>[0-9]+)/(?P<scan_id>t[0-9]+)/[0-9]+_t1w_deface_stx/image-slice(?P<slice_id>[0-9]+).jpg",
        input_path)[0]
      target_mask_paths = glob.glob(os.path.join(prefix, f"Site{site_id}/{patient_id}/{scan_id}/{patient_id}_LesionSmooth_*/image-slice{slice_id}.jpg"), recursive=True)
      for target_mask_path in target_mask_paths:
        dev_input_paths.append(input_path)
        dev_target_mask_paths.append(target_mask_path)
  elif FLAGS.split_type == "by_site":
    pass  # TODO
  else:
    raise ValueError(f"Specified unknown FLAGS.split_type={FLAGS.split_type}.")

  return (
    train_input_paths,
    train_target_mask_paths,
    dev_input_paths,
    dev_target_mask_paths,
  )

