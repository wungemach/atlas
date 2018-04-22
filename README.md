# ATLAS
```bash
$ git clone --recurse-submodules https://github.com/gnedivad/atlas
```

## Helpful flags for sanity checking
- `--use_fake_target_masks`: sets all target masks to entirely zeros, a label that all models should be able to learn. In this example, I verify that model `ZeroATLASModel`, which predicts 0 for all masks, achieves a low loss.
  ```bash
  $ python main.py --experiment_name=zero --model_name=ZeroATLASModel --use_fake_target_masks
  ```

- `--input_regex`: sets the regex to use for input paths. If set, `FLAGS.p` will be ignored and train and dev sets will use this same input regex. For small enough sets, all models should be able to overfit the target masks. In this example, I verify that the baseline model, can overfit to the lesions for a single slice.
  ```bash
  $ python main.py --experiment_name=regex --input_regex=Site2/031844/t01/031844_t1w_deface_stx/image-slice102.jpg
  ```

## Helpful flags to toggle
- `merge_target_masks`: merges target masks, if more than one exists, for each slice into a single mask.
  ```bash
  $ python main.py --experiment_name=merge --input_regex=Site2/031844/t01/031844_t1w_deface_stx/image-slice102.jpg --merge_target_masks
