import os
import re
import glob
import sys
import numpy as np
import uproot
import awkward as ak


# ============================================================
# fill paths here
# ============================================================

# background logs: used to compute sigma_b
BKG_LOG_PATTERN = "/eos/user/y/yzhang4/LIV/LO/output/logs/*.log"

# signal logs: used together with one reconstructed ROOT file
SIG_LOG_PATTERN = "/eos/user/y/yzhang4/LIV/LO/output/logs/*.log"

# reconstructed ROOT file prefix directory
RECONSTRUCTED_DIR = "/eos/user/y/yzhang4/LIV/Split-theta/theta-split-5/files-reconstructed"


# ============================================================
# regex patterns
# ============================================================

XS_PATTERN = re.compile(
    r"Cross-section\s*:\s*([0-9eE.+-]+)\s*\+\-\s*([0-9eE.+-]+)\s*pb"
)
NEVT_PATTERN = re.compile(
    r"Nb of events\s*:\s*([0-9eE.+-]+)"
)


def parse_log_file(path):
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read()

    xs_match = XS_PATTERN.search(text)
    nevt_match = NEVT_PATTERN.search(text)

    if xs_match is None:
        return None

    xs = float(xs_match.group(1))
    err = float(xs_match.group(2))
    nevt = float(nevt_match.group(1)) if nevt_match is not None else np.nan

    return {
        "file": path,
        "xs_pb": xs,
        "err_pb": err,
        "n_events": nevt,
    }


def build_signal_root_file(theta_tag):
    return os.path.join(
        RECONSTRUCTED_DIR,
        f"Reconstructed_Z1Z2_LHEF_{theta_tag}.root"
    )


def load_reweight_from_root(file_path, tree_name="LHEF", epsilon=1e-12, max_weight=1e6):
    tree = uproot.open(file_path)[tree_name]

    pt = tree["Particle.PT"].array()
    sm_amp = tree["SM_Amplitude"].array()
    nc_amp = tree["NC_Amplitude"].array()

    mask = ak.num(pt) > 7

    safe_sm_amp = ak.to_numpy(sm_amp[mask])
    safe_nc_amp = ak.to_numpy(nc_amp[mask])

    ratio = np.zeros(len(safe_nc_amp), dtype=float)
    nonzero_mask = np.abs(safe_sm_amp) > epsilon
    ratio[nonzero_mask] = safe_nc_amp[nonzero_mask] / safe_sm_amp[nonzero_mask]

    w = 1.0 + ratio
    w = np.clip(w, 0.0, max_weight)

    valid = np.isfinite(w)
    w = w[valid]

    if len(w) == 0:
        raise RuntimeError(f"No valid reweights found in ROOT file: {file_path}")

    return w


def weighted_mean(values, errors):
    values = np.asarray(values, dtype=float)
    errors = np.asarray(errors, dtype=float)

    valid = np.isfinite(values) & np.isfinite(errors) & (errors > 0)
    if not np.any(valid):
        raise RuntimeError("No valid values with positive uncertainties for weighted mean.")

    w = 1.0 / errors[valid] ** 2
    mean = np.sum(w * values[valid]) / np.sum(w)
    err = np.sqrt(1.0 / np.sum(w))
    return mean, err


def collect_background_sigma():
    log_files = sorted(glob.glob(BKG_LOG_PATTERN))
    if not log_files:
        raise FileNotFoundError(f"No background log files matched: {BKG_LOG_PATTERN}")

    parsed = []
    for path in log_files:
        info = parse_log_file(path)
        if info is not None:
            parsed.append(info)

    if not parsed:
        raise RuntimeError("No valid background cross-sections found.")

    xs = [x["xs_pb"] for x in parsed]
    errs = [x["err_pb"] for x in parsed]
    sigma_b, sigma_b_err = weighted_mean(xs, errs)

    print("\n=== BACKGROUND SUMMARY ===")
    print(f"n_files              = {len(parsed)}")
    print(f"simple_mean_xs_pb    = {np.mean(xs):.10f}")
    print(f"weighted_mean_xs_pb  = {sigma_b:.10f}")
    print(f"weighted_mean_err_pb = {sigma_b_err:.10f}")

    return sigma_b, sigma_b_err


def collect_signal_sigma_for_theta(theta_tag):
    log_files = sorted(glob.glob(SIG_LOG_PATTERN))
    if not log_files:
        raise FileNotFoundError(f"No signal log files matched: {SIG_LOG_PATTERN}")

    root_file = build_signal_root_file(theta_tag)
    if not os.path.exists(root_file):
        raise FileNotFoundError(f"Reconstructed ROOT file not found: {root_file}")

    parsed = []
    for path in log_files:
        info = parse_log_file(path)
        if info is not None:
            parsed.append(info)

    if not parsed:
        raise RuntimeError("No valid signal cross-sections found.")

    xs = [x["xs_pb"] for x in parsed]
    errs = [x["err_pb"] for x in parsed]
    sigma_k, sigma_k_err = weighted_mean(xs, errs)

    w = load_reweight_from_root(root_file)
    mean_w = np.mean(w)
    std_w = np.std(w, ddof=1) if len(w) > 1 else 0.0

    sigma_sbi = sigma_k * mean_w
    sigma_sbi_err = sigma_k_err * mean_w

    print("\n=== SIGNAL SUMMARY ===")
    print(f"theta_tag            = {theta_tag}")
    print(f"root_file            = {root_file}")
    print(f"weighted_mean_xs_pb  = {sigma_k:.10f}")
    print(f"weighted_mean_err_pb = {sigma_k_err:.10f}")
    print(f"mean_reweight        = {mean_w:.10f}")
    print(f"std_reweight         = {std_w:.10f}")
    print(f"sigma_sbi_pb         = {sigma_sbi:.10f}")
    print(f"sigma_sbi_err_pb     = {sigma_sbi_err:.10f}")

    return sigma_sbi, sigma_sbi_err


def main(theta_tag):
    sigma_b, sigma_b_err = collect_background_sigma()
    sigma_sbi0, sigma_sbi0_err = collect_signal_sigma_for_theta(theta_tag)

    sigma_i0 = sigma_sbi0 - sigma_b

    print("\n=== FINAL VALUES ===")
    print(f"sigma_b    = {sigma_b:.10f}  # pb")
    print(f"sigma_sbi0 = {sigma_sbi0:.10f}  # pb")
    print(f"sigma_i0   = {sigma_i0:.10f}  # pb")

    print("\n=== MACHINE READABLE ===")
    print(f"SIGMA_B={sigma_b:.10f}")
    print(f"SIGMA_SBI0={sigma_sbi0:.10f}")
    print(f"SIGMA_I0={sigma_i0:.10f}")

    print("\n=== PASTE INTO YOUR ANALYSIS SCRIPT ===")
    print(f"sigma_b = {sigma_b:.10f}")
    print(f"sigma_sbi0 = {sigma_sbi0:.10f}")
    print("sigma_i0 = sigma_sbi0 - sigma_b")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python3 extract_sigmas.py <theta_tag>")
        sys.exit(1)

    theta_tag = sys.argv[1]
    main(theta_tag)
