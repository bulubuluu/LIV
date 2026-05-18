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

## Asimov dataset and likelihood in `1.py`

This part of `1.py` builds an Asimov-like observed sample from the ROOT input, then evaluates the shape term, rate term, and total likelihood scan.

### 1. Build the Asimov-like dataset

The ROOT file is first read with:

```python
X_obs, W_obs = dm.load_root_features(obs_root)
```

The event weights are then normalized and rescaled to the expected yield:

```python
W_asimov = W_obs / np.sum(W_obs)
W_asimov = W_asimov * lumi * (sigma_sbi0 * 2.5 + sigma_b)

events_obs = RootProcess(
    kinematics=pd.DataFrame(X_obs, columns=FEATURES),
    weights=pd.Series(W_asimov),
)
```

This corresponds to building an expected weighted dataset with total normalization

```math
N_{\mathrm{Asimov}} = \mathcal{L}\left(\sigma_{\mathrm{SBI},0}\cdot 2.5 + \sigma_{\mathrm{B}}\right),
```

where:

- `lumi` is the integrated luminosity
- `sigma_sbi0` is the reference SBI cross section
- `sigma_b` is the background cross section

In the code, the factor `2.5` is the reference signal-strength-like multiplier used to build the Asimov sample.

### 2. Reference likelihood ratio

The trained CARL model gives the reference density ratio

```math
r_{\mathrm{ref}}(x) = \frac{p_{\mathrm{SBI}}(x \mid \theta_0)}{p_{\mathrm{B}}(x)}.
```

In code:

```python
r_ref_obs = get_likelihood_ratio(events_obs, FEATURES, scaler, model)
r_ref_obs = torch.clamp(r_ref_obs, min=1e-6, max=1e6)
```

### 3. Likelihood ratio at scanned `theta`

The scan is performed in terms of

```math
\lambda = \frac{\theta}{\theta_0},
```

with:

```python
lambda_space = torch.linspace(-5.0, 5.0, 501)
theta_space = lambda_space * theta0
```

The code uses

```math
r(x \mid \theta) =
\frac{\lambda\,\sigma_{\mathrm{SBI},0}\,r_{\mathrm{ref}}(x)
      + \sigma_{\mathrm{B}}(1-\lambda)}
     {\lambda\,\sigma_{\mathrm{I},0} + \sigma_{\mathrm{B}}}.
```

In code:

```python
r_theta = (
    lambda_space[None, :] * sigma_sbi0 * r_ref_obs[:, None]
    + sigma_b * (1.0 - lambda_space[None, :])
) / (
    lambda_space[None, :] * sigma_i0 + sigma_b
)

r_theta = torch.clamp(r_theta, min=1e-6, max=1e6)
```

Here:

```math
\sigma_{\mathrm{I},0} = \sigma_{\mathrm{SBI},0} - \sigma_{\mathrm{B}}.
```

### 4. Shape term

The shape contribution to the test statistic is computed from the weighted log-likelihood ratio:

```math
t_{\mathrm{shape}}(\theta)
= -2 \sum_i w_i \log r(x_i \mid \theta).
```

In code:

```python
t_shape = -2.0 * torch.sum(
    w_obs[:, None] * torch.log(r_theta),
    dim=0
)
```

### 5. Rate term

The expected event yield for each scan point is:

```math
\nu(\theta) = \mathcal{L}\left(\lambda\,\sigma_{\mathrm{I},0} + \sigma_{\mathrm{B}}\right),
```

with the background-only reference:

```math
\nu_{\mathrm{ref}} = \mathcal{L}\sigma_{\mathrm{B}}.
```

The Poisson negative log-likelihood used in the code is:

```math
-\log \mathcal{L}_{\mathrm{Pois}}(n \mid \nu) = \nu - n\log\nu.
```

implemented as:

```python
def neg_log_pois(n_obs, nu_exp, eps=1e-12):
    nu_exp = torch.clamp(nu_exp, min=eps)
    return nu_exp - n_obs * torch.log(nu_exp)
```

The rate test statistic is then:

```math
t_{\mathrm{rate}}(\theta)
= 2\left[
-\log \mathcal{L}_{\mathrm{Pois}}(n_{\mathrm{obs}} \mid \nu(\theta))
+
\log \mathcal{L}_{\mathrm{Pois}}(n_{\mathrm{obs}} \mid \nu_{\mathrm{ref}})
\right].
```

In code:

```python
n_obs = torch.sum(w_obs)
nu_theta = lumi * (lambda_space * sigma_i0 + sigma_b)
nu_ref = torch.full_like(nu_theta, lumi * sigma_b)

t_rate = 2.0 * (
    neg_log_pois(n_obs, nu_theta) - neg_log_pois(n_obs, nu_ref)
)
```

### 6. Total test statistic

The final scan combines shape and rate:

```math
t_{\mathrm{total}}(\theta) = t_{\mathrm{shape}}(\theta) + t_{\mathrm{rate}}(\theta).
```

In code:

```python
t_total = t_rate + t_shape
```

The script then saves plots for:

- `t_shape`
- `t_rate`
- `t_total`
- scans in both `theta` and `lambda`
- overlay plots comparing all three contributions

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
