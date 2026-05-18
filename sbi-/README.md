This directory is used to train a CARL model with reconstructed `Z1` and `Z2` observables, and then run the plotting / analysis step with `1.py`.

## Environment

```bash
cd /eos/user/y/yzhang4/LIV/LO/
source venv/bin/activate
cd /eos/user/y/yzhang4/LIV/sbi-/sessions/day2/nsbi-tutorial
```

## Input and output

- Input ROOT file:
  `/eos/user/y/yzhang4/LIV/split-Z/files/Reconstructed_Z1Z2_LHEF_4e-6.root`
- Training output directory:
  `run/liv_over_bkg/`

## Training

Run:

```bash
cd /eos/user/y/yzhang4/LIV/LO/
source venv/bin/activate
cd /eos/user/y/yzhang4/LIV/sbi-/sessions/day2/nsbi-tutorial

python3 -m nsbi.carl fit \
  --data.numerator_events /eos/user/y/yzhang4/LIV/split-Z/files/Reconstructed_Z1Z2_LHEF_4e-6.root \
  --data.features '["Z1_PT","Z1_Eta","Z1_Phi","Z1_Mass","Z2_PT","Z2_Eta","Z2_Phi","Z2_Mass"]' \
  --data.batch_size 1024 \
  --data.data_dir run/liv_over_bkg \
  --model.learning_rate 1e-5 \
  --model.n_layers 16 \
  --model.n_nodes 1024 \
  --trainer.devices 1 \
  --trainer.max_epochs 500 \
  --trainer.accelerator cpu
```

## Plotting

Run:

```bash
cd /eos/user/y/yzhang4/LIV/LO/
source venv/bin/activate
cd /eos/user/y/yzhang4/LIV/sbi-/sessions/day2/nsbi-tutorial

python3 1.py
```

## Workflow of `1.py`

`1.py` is the main analysis and plotting script. Its workflow is:

1. Load the trained datasets, scaler, and trained model from `run/<run_name>/`.
2. Evaluate the likelihood ratio at the reference point `theta0`.
3. Perform closure checks on validation events:
   - 1D closure plots such as `Z1_PT` and `Z1_Mass`
   - 2D closure plots such as `Z1_Eta` vs `Z1_Phi` and `Z2_Eta` vs `Z2_Phi`
4. Load the observed ROOT file and build the weighted observed / Asimov-like event sample.
5. Evaluate the likelihood ratio for the observed events.
6. Scan over `lambda` and `theta` to compute:
   - shape term
   - rate term
   - total test statistic
7. Save all plots and the scan summary under:
   `run/debug_root/analysis_theta_scan/`

Main outputs produced by `1.py` include:

- closure plots
- `t_shape`, `t_rate`, and `t_total` scan plots
- sigma-line versions of the scan plots
- overlay plots comparing shape / rate / total contributions
- `summary.txt`

## Important files

- `1.py`
  Main plotting / analysis entry point.

- `nsbi/carl/__main__.py`
  Training entry point for `python3 -m nsbi.carl fit`.

- `nsbi/carl/utils.py`
  Helper functions for loading saved models, scalers, and training results.

- `run/liv_over_bkg/`
  Stores training outputs such as logs, checkpoints, and saved preprocessing objects.

- `/eos/user/y/yzhang4/LIV/split-Z/files/Reconstructed_Z1Z2_LHEF_4e-6.root`
  Input dataset used for training.

## File roles

- Training command
  Defines the input ROOT file, selected features, output directory, and training hyperparameters.

- `1.py`
  Main plotting and post-processing script. Handles closure checks, observed sample construction, parameter scan, and output plots.

- `nsbi/carl/utils.py`
  Utility functions for loading saved datasets, scalers, checkpoints, and likelihood-ratio predictions.

- `nsbi/carl/__main__.py`
  CLI entry point for training with `python3 -m nsbi.carl fit`.

- `run/liv_over_bkg/`
  Training output directory containing logs, saved preprocessing objects, and trained checkpoints.

- `/eos/user/y/yzhang4/LIV/split-Z/files/Reconstructed_Z1Z2_LHEF_4e-6.root`
  Main training input dataset.
