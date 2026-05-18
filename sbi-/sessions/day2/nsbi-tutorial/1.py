import os
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

from datasets.balanced import RootProcess, BalancedDataModule
from nsbi.carl.utils import load_data, load_results, get_likelihood_ratio


FEATURES = ['Z1_PT', 'Z1_Eta', 'Z1_Phi', 'Z1_Mass',
            'Z2_PT', 'Z2_Eta', 'Z2_Phi', 'Z2_Mass']


def neg_log_pois(n_obs, nu_exp, eps=1e-12):
    nu_exp = torch.clamp(nu_exp, min=eps)
    return nu_exp - n_obs * torch.log(nu_exp)


def weighted_hist(x, w, bins):
    h, edges = np.histogram(x, bins=bins, weights=w)
    return h, edges


def make_closure_plot(events_num, events_den, r_den_to_num, observable, outpath, nbins=30):
    x_num = events_num.kinematics[observable].to_numpy()
    x_den = events_den.kinematics[observable].to_numpy()

    w_num = events_num.weights.to_numpy()
    w_den = events_den.weights.to_numpy()
    w_den_to_num = w_den * r_den_to_num.detach().cpu().numpy()

    xmin = min(x_num.min(), x_den.min())
    xmax = max(x_num.max(), x_den.max())
    bins = np.linspace(xmin, xmax, nbins + 1)

    h_num, edges = weighted_hist(x_num, w_num, bins)
    h_den_to_num, _ = weighted_hist(x_den, w_den_to_num, bins)

    centers = 0.5 * (edges[:-1] + edges[1:])
    ratio = np.divide(
        h_den_to_num,
        h_num,
        out=np.zeros_like(h_num, dtype=float),
        where=h_num != 0,
    )

    fig, (ax, rax) = plt.subplots(
        2, 1, figsize=(7, 6), sharex=True,
        gridspec_kw={"height_ratios": [3, 1]}
    )

    ax.step(centers, h_num, where="mid", label="SBI", linewidth=2)
    ax.step(centers, h_den_to_num, where="mid", label="B x r(x)", linewidth=2)
    ax.set_ylabel("Weighted counts")
    ax.set_title(f"Closure check: {observable}")
    ax.legend()

    rax.axhline(1.0, color="black", linestyle="--", linewidth=1)
    rax.plot(centers, ratio, marker="o", linestyle="-")
    rax.set_xlabel(observable)
    rax.set_ylabel("ratio")
    rax.set_ylim(0.8, 1.2)

    fig.tight_layout()
    fig.savefig(outpath, dpi=150)
    plt.close(fig)

    return ratio


def plot_scan(theta_space, t, outpath, ylabel, ymin=None, ymax=None, draw_sigma_lines=False):
    t = t.clone()
    t -= torch.min(t)
    theta_fit = theta_space[torch.argmin(t)]

    plt.figure(figsize=(7, 5))
    plt.plot(theta_space.cpu().numpy(), t.cpu().numpy(), linewidth=2)
    plt.scatter(theta_fit.cpu().item(), 0.0, color="red", label=f"best fit = {theta_fit:.3e}")

    if draw_sigma_lines:
        plt.axhline(1.0, color="gray", linestyle="--", label="1σ")
        plt.axhline(4.0, color="gray", linestyle="--", label="2σ")
        plt.axhline(9.0, color="gray", linestyle="--", label="3σ")

    if ymin is not None or ymax is not None:
        plt.ylim(ymin, ymax)

    plt.xlabel("theta")
    plt.ylabel(ylabel)
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close()


def make_closure_plot_2d(
    events_num,
    events_den,
    r_den_to_num,
    observable_x,
    observable_y,
    outpath,
    nbins_x=30,
    nbins_y=30,
    xlim=None,
    ylim=None,
):
    x_num = events_num.kinematics[observable_x].to_numpy()
    y_num = events_num.kinematics[observable_y].to_numpy()
    x_den = events_den.kinematics[observable_x].to_numpy()
    y_den = events_den.kinematics[observable_y].to_numpy()

    w_num = events_num.weights.to_numpy()
    w_den = events_den.weights.to_numpy()
    w_den_to_num = w_den * r_den_to_num.detach().cpu().numpy()

    if xlim is None:
        x_min = min(x_num.min(), x_den.min())
        x_max = max(x_num.max(), x_den.max())
    else:
        x_min, x_max = xlim

    if ylim is None:
        y_min = min(y_num.min(), y_den.min())
        y_max = max(y_num.max(), y_den.max())
    else:
        y_min, y_max = ylim

    x_bins = np.linspace(x_min, x_max, nbins_x + 1)
    y_bins = np.linspace(y_min, y_max, nbins_y + 1)

    h_num, x_edges, y_edges = np.histogram2d(
        x_num, y_num, bins=[x_bins, y_bins], weights=w_num
    )
    h_den, _, _ = np.histogram2d(
        x_den, y_den, bins=[x_bins, y_bins], weights=w_den
    )
    h_den_to_num, _, _ = np.histogram2d(
        x_den, y_den, bins=[x_bins, y_bins], weights=w_den_to_num
    )

    ratio = np.divide(
        h_den_to_num,
        h_num,
        out=np.zeros_like(h_num, dtype=float),
        where=h_num != 0,
    )

    common_vmin = min(
        np.nanmin(h_num),
        np.nanmin(h_den),
        np.nanmin(h_den_to_num),
    )
    common_vmax = max(
        np.nanmax(h_num),
        np.nanmax(h_den),
        np.nanmax(h_den_to_num),
    )

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    im0 = axes[0, 0].pcolormesh(
        x_edges, y_edges, h_num.T, shading="auto",
        vmin=common_vmin, vmax=common_vmax
    )
    axes[0, 0].set_title(f"SBI: {observable_x} vs {observable_y}")
    axes[0, 0].set_xlabel(observable_x)
    axes[0, 0].set_ylabel(observable_y)
    axes[0, 0].set_xlim(x_min, x_max)
    axes[0, 0].set_ylim(y_min, y_max)
    plt.colorbar(im0, ax=axes[0, 0])

    im1 = axes[0, 1].pcolormesh(
        x_edges, y_edges, h_den.T, shading="auto",
        vmin=common_vmin, vmax=common_vmax
    )
    axes[0, 1].set_title(f"B: {observable_x} vs {observable_y}")
    axes[0, 1].set_xlabel(observable_x)
    axes[0, 1].set_ylabel(observable_y)
    axes[0, 1].set_xlim(x_min, x_max)
    axes[0, 1].set_ylim(y_min, y_max)
    plt.colorbar(im1, ax=axes[0, 1])

    im2 = axes[1, 0].pcolormesh(
        x_edges, y_edges, h_den_to_num.T, shading="auto",
        vmin=common_vmin, vmax=common_vmax
    )
    axes[1, 0].set_title(f"B x r(x): {observable_x} vs {observable_y}")
    axes[1, 0].set_xlabel(observable_x)
    axes[1, 0].set_ylabel(observable_y)
    axes[1, 0].set_xlim(x_min, x_max)
    axes[1, 0].set_ylim(y_min, y_max)
    plt.colorbar(im2, ax=axes[1, 0])

    im3 = axes[1, 1].pcolormesh(
        x_edges, y_edges, ratio.T, shading="auto",
        vmin=0.8, vmax=1.2, cmap="coolwarm"
    )
    axes[1, 1].set_title(f"Closure ratio: {observable_x} vs {observable_y}")
    axes[1, 1].set_xlabel(observable_x)
    axes[1, 1].set_ylabel(observable_y)
    axes[1, 1].set_xlim(x_min, x_max)
    axes[1, 1].set_ylim(y_min, y_max)
    plt.colorbar(im3, ax=axes[1, 1])

    fig.tight_layout()
    fig.savefig(outpath, dpi=150)
    plt.close(fig)

    return ratio

def plot_scan_lambda(lambda_space, t, outpath, ylabel, ymin=None, ymax=None,
                     draw_sigma_lines=False, logy=False):
    t = t.clone()
    t -= torch.min(t)

    x = lambda_space.detach().cpu().numpy()
    y = t.detach().cpu().numpy()

    best_idx = np.argmin(y)
    lambda_fit_value = x[best_idx]
    t_fit_value = y[best_idx]

    if logy:
        y = y + 1e-6
        t_fit_value = t_fit_value + 1e-6

    plt.figure(figsize=(7, 5))
    plt.plot(x, y, linewidth=1.2, alpha=0.7, color="C0")
    plt.scatter(x, y, s=16, color="C0", label="scan points")
    plt.scatter(lambda_fit_value, t_fit_value, color="red", s=50, zorder=3,
                label=f"best fit = {lambda_fit_value:.3e}")

    if draw_sigma_lines:
        plt.axhline(1.0, color="gray", linestyle="--", label="1σ")
        plt.axhline(4.0, color="gray", linestyle="--", label="2σ")
        plt.axhline(9.0, color="gray", linestyle="--", label="3σ")

    if logy:
        plt.yscale("log")

    if ymin is not None or ymax is not None:
        plt.ylim(bottom=ymin, top=ymax)

    plt.xlabel("lambda")
    plt.ylabel(ylabel)
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close()


def main():
    run_dir = "run"
    run_name = "debug_root"
    out_dir = os.path.join(run_dir, run_name, "analysis_theta_scan")
    os.makedirs(out_dir, exist_ok=True)

    # ============================================================
    # user inputs
    # ============================================================
    theta0 = 4.0e-6
    # theta_space = torch.linspace(-5.0e-5, 5.0e-5, 501)
    # lambda_space = theta_space / theta0
    lambda_space = torch.linspace(-5.0, 5.0, 501)
    theta_space = lambda_space * theta0

    # observed ROOT file
    # obs_root = "/eos/user/y/yzhang4/LIV/split-Z/files/Reconstructed_Z1Z2_LHEF_1e-5_sample_1.root"
    obs_root = "/eos/user/y/yzhang4/LIV/split-Z/files/Reconstructed_Z1Z2_LHEF_1e-5.root"
    
    # placeholders: replace with your physics inputs
    sigma_b = 0.0104701189
    sigma_sbi0 = 0.0104739004
    sigma_i0 = sigma_sbi0 - sigma_b
    lumi = 1153291.6866890918  # pb^-1 假设一下DATA 数据

    # ============================================================
    # load trained objects
    # ============================================================
    (events_num_train, events_num_val), (events_den_train, events_den_val) = load_data(run_dir, run_name)
    scaler, model = load_results(run_dir, run_name)

    # ============================================================
    # step 3: closure check at reference theta0
    # ============================================================
    r_num_ref = get_likelihood_ratio(events_num_val, FEATURES, scaler, model)
    r_den_ref = get_likelihood_ratio(events_den_val, FEATURES, scaler, model)

    r_num_ref = torch.clamp(r_num_ref, min=1e-6, max=1e6)
    r_den_ref = torch.clamp(r_den_ref, min=1e-6, max=1e6)

    print("=== Basic ratio info at reference theta0 ===")
    print("mean r_num_ref =", torch.mean(r_num_ref).item())
    print("mean r_den_ref =", torch.mean(r_den_ref).item())
    print("median r_num_ref =", torch.median(r_num_ref).item())
    print("median r_den_ref =", torch.median(r_den_ref).item())

    ratio_Z1_PT = make_closure_plot(
        events_num_val,
        events_den_val,
        r_den_ref,
        observable="Z1_PT",
        outpath=os.path.join(out_dir, "closure_Z1_PT.png"),
        nbins=30,
    )
    ratio_Z1_Mass = make_closure_plot(
        events_num_val,
        events_den_val,
        r_den_ref,
        observable="Z1_Mass",
        outpath=os.path.join(out_dir, "closure_Z1_Mass.png"),
        nbins=30,
    )

    print("Z1_PT closure ratio first 10 bins   =", ratio_Z1_PT[:10])
    print("Z1_Mass closure ratio first 10 bins =", ratio_Z1_Mass[:10])

    ratio_Z1_Eta_Z1_Phi = make_closure_plot_2d(
        events_num_val,
        events_den_val,
        r_den_ref,
        observable_x="Z1_Eta",
        observable_y="Z1_Phi",
        outpath=os.path.join(out_dir, "closure_2d_Z1_Phi_Z1_Eta.png"),
        nbins_y=50,
        nbins_x=50,
        xlim=(-3, 3),
        ylim=(-4, 4),
    )

    ratio_Z2_Eta_Z2_Phi = make_closure_plot_2d(
        events_num_val,
        events_den_val,
        r_den_ref,
        observable_x="Z2_Eta",
        observable_y="Z2_Phi",
        outpath=os.path.join(out_dir, "closure_2d_phi4_eta4.png"),
        nbins_x=30,
        nbins_y=30,
        xlim=(-4, 4),
        ylim=(-3, 3),
    )



    # ============================================================
    # build observed data
    # data weights = 1
    # ============================================================
    dm = BalancedDataModule(
        numerator_events=obs_root,
        features=FEATURES,
        data_dir=out_dir,
    )

    # X_obs, _ = dm.load_root_features(obs_root)

    # events_obs = RootProcess(
    #     kinematics=pd.DataFrame(X_obs, columns=FEATURES),
    #     weights=pd.Series(np.ones(len(X_obs))),
    # )

    # 先用MC测试一下
    # X_obs, W_obs = dm.load_root_features(obs_root)

    # events_obs = RootProcess(
    #     kinematics=pd.DataFrame(X_obs, columns=FEATURES),
    #     weights=pd.Series(W_obs),
    # )
    
    X_obs, W_obs = dm.load_root_features(obs_root)

    W_asimov = W_obs / np.sum(W_obs)
    W_asimov = W_asimov * lumi * (sigma_sbi0 * 2.5 + sigma_b)

    events_obs = RootProcess(
        kinematics=pd.DataFrame(X_obs, columns=FEATURES),
        weights=pd.Series(W_asimov),
    )

    w_obs = torch.tensor(events_obs.weights.to_numpy(), dtype=torch.float32)


    # ============================================================
    # step 4: theta scan
    #
    # r_ref(x) = p_SBI(x | theta0) / p_B(x)
    #
    # r(x | theta) =
    # [lambda * sigma_sbi0 * r_ref(x) + sigma_b * (1 - lambda)]
    # / [lambda * sigma_i0 + sigma_b]
    # ============================================================
    r_ref_obs = get_likelihood_ratio(events_obs, FEATURES, scaler, model)
    r_ref_obs = torch.clamp(r_ref_obs, min=1e-6, max=1e6)

    r_theta = (
        lambda_space[None, :] * sigma_sbi0 * r_ref_obs[:, None]
        + sigma_b * (1.0 - lambda_space[None, :])
    ) / (
        lambda_space[None, :] * sigma_i0 + sigma_b
    )

    r_theta = torch.clamp(r_theta, min=1e-6, max=1e6)

    t_shape = -2.0 * torch.sum(
        w_obs[:, None] * torch.log(r_theta),
        dim=0
    )

    n_obs = torch.sum(w_obs)
    nu_theta = lumi * (lambda_space * sigma_i0 + sigma_b)
    nu_ref = torch.full_like(nu_theta, lumi * sigma_b)

    t_rate = 2.0 * (
        neg_log_pois(n_obs, nu_theta) - neg_log_pois(n_obs, nu_ref)
    )

    t_total = t_rate + t_shape

    print("\n=== Theta scan summary ===")
    print("n_obs =", n_obs.item())
    print("theta best fit (shape) =", theta_space[torch.argmin(t_shape)].item())
    print("theta best fit (rate)  =", theta_space[torch.argmin(t_rate)].item())
    print("theta best fit (total) =", theta_space[torch.argmin(t_total)].item())

    plot_scan(theta_space, t_shape, os.path.join(out_dir, "t_shape_scan.png"), ylabel="t_shape")
    plot_scan(theta_space, t_rate, os.path.join(out_dir, "t_rate_scan.png"), ylabel="t_rate")
    plot_scan(theta_space, t_total, os.path.join(out_dir, "t_total_scan.png"), ylabel="t_total")

    plot_scan_lambda(lambda_space, t_shape, os.path.join(out_dir, "lambda_t_shape_scan.png"), ylabel="t_shape", logy=False)
    plot_scan_lambda(lambda_space, t_rate, os.path.join(out_dir, "lambda_t_rate_scan.png"), ylabel="t_rate", logy=False)
    plot_scan_lambda(lambda_space, t_total, os.path.join(out_dir, "lambda_t_total_scan.png"), ylabel="t_total", logy=False)

    plot_scan(
        theta_space,
        t_total,
        os.path.join(out_dir, "t_total_scan_sigma.png"),
        ylabel="t_total",
        ymin=-1,
        ymax=10,
        draw_sigma_lines=True,
    )
    
    plot_scan_lambda(
        lambda_space,
        t_total,
        os.path.join(out_dir, "lambda_t_total_scan_sigma.png"),
        ylabel="t_total",
        # ymin=-1,
        # ymax=300,
        draw_sigma_lines=True,
        logy=True
    )

    plt.figure(figsize=(7,5))
    plt.plot(theta_space.cpu(), (t_shape - torch.min(t_shape)).cpu(), label="shape")
    plt.plot(theta_space.cpu(), (t_rate - torch.min(t_rate)).cpu(), label="rate")
    plt.plot(theta_space.cpu(), (t_total - torch.min(t_total)).cpu(), label="total")
    plt.legend()
    plt.xlabel("theta")
    plt.ylabel("shifted test statistic")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "t_components_overlay.png"), dpi=150)
    plt.close()

    plt.figure(figsize=(7,5))
    plt.plot(theta_space.cpu(), (t_shape - torch.min(t_shape)).cpu(), label="shape")
    plt.plot(theta_space.cpu(), (t_rate - torch.min(t_rate)).cpu(), label="rate")
    plt.plot(theta_space.cpu(), (t_total - torch.min(t_total)).cpu(), label="total")
    plt.axhline(1.0, color="grey", linestyle="--")
    plt.axhline(4.0, color="grey", linestyle="--")
    plt.axhline(9.0, color="grey", linestyle="--")
    plt.ylim(-1, 10)
    plt.legend()
    plt.xlabel("theta")
    plt.ylabel("shifted test statistic")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "t_components_overlay-3sigma.png"), dpi=150)
    plt.close()

    with open(os.path.join(out_dir, "summary.txt"), "w") as f:
        f.write("Theta scan summary\n")
        f.write(f"obs_root = {obs_root}\n")
        f.write(f"theta0 = {theta0}\n")
        f.write(f"sigma_b = {sigma_b}\n")
        f.write(f"sigma_i0 = {sigma_i0}\n")
        f.write(f"sigma_sbi0 = {sigma_sbi0}\n")
        f.write(f"lumi = {lumi}\n")
        f.write(f"n_obs = {n_obs.item()}\n")
        f.write(f"best theta shape = {theta_space[torch.argmin(t_shape)].item()}\n")
        f.write(f"best theta rate  = {theta_space[torch.argmin(t_rate)].item()}\n")
        f.write(f"best theta total = {theta_space[torch.argmin(t_total)].item()}\n")
        f.write(f"Z1_PT closure first 10 bins = {ratio_Z1_PT[:10]}\n")
        f.write(f"Z1_Mass closure first 10 bins = {ratio_Z1_Mass[:10]}\n")

    print(f"\nSaved outputs to: {out_dir}")


if __name__ == "__main__":
    main()
