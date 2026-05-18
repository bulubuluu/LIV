import os
import numpy as np
import uproot
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def weighted_mean_std(x, w=None):
    x = np.asarray(x)

    if w is None:
        return np.mean(x), np.std(x)

    w = np.asarray(w)
    wsum = np.sum(w)

    if wsum == 0:
        return np.nan, np.nan

    mean = np.sum(w * x) / wsum
    var = np.sum(w * (x - mean) ** 2) / wsum

    if var < 0:
        std = np.nan
    else:
        std = np.sqrt(var)

    return mean, std


def add_2d_stats_box(ax, name, x, y, weights=None):
    entries = len(x)

    mean_x, std_x = weighted_mean_std(x, weights)
    mean_y, std_y = weighted_mean_std(y, weights)

    text = (
        f"{name}\n"
        f"Entries      {entries:d}\n"
        f"Mean x   {mean_x: .4g}\n"
        f"Mean y   {mean_y: .4g}\n"
        f"Std Dev x {std_x: .4g}\n"
        f"Std Dev y {std_y: .4g}"
    )

    ax.text(
        0.98,
        0.98,
        text,
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=8,
        bbox=dict(
            facecolor="white",
            edgecolor="black",
            boxstyle="square,pad=0.35",
        ),
    )


def get_common_range(x_sm, y_sm, x_liv, y_liv):
    return (
        min(np.min(x_sm), np.min(y_sm), np.min(x_liv), np.min(y_liv)),
        max(np.max(x_sm), np.max(y_sm), np.max(x_liv), np.max(y_liv)),
    )


def plot_eta_phi_heatmap_pair(
    eta_sm,
    phi_sm,
    eta_liv,
    phi_liv,
    liv_weight,
    z_label,
    output_path,
    bins=60,
    use_same_color_scale=False,
    theta_tag=None,
):
    # eta_range = (
    #     min(np.min(eta_sm), np.min(eta_liv)),
    #     max(np.max(eta_sm), np.max(eta_liv)),
    # )

    # phi_range = (
    #     min(np.min(phi_sm), np.min(phi_liv)),
    #     max(np.max(phi_sm), np.max(phi_liv)),
    # )
    eta_range = (-4, 4)
    phi_range = (-3, 3)

    h_sm, xedges, yedges = np.histogram2d(
        eta_sm,
        phi_sm,
        bins=bins,
        range=[eta_range, phi_range],
    )

    h_liv, _, _ = np.histogram2d(
        eta_liv,
        phi_liv,
        bins=[xedges, yedges],
        weights=liv_weight,
    )

    if use_same_color_scale:
        vmin = min(np.nanmin(h_sm), np.nanmin(h_liv))
        vmax = max(np.nanmax(h_sm), np.nanmax(h_liv))
        sm_vmin, sm_vmax = vmin, vmax
        liv_vmin, liv_vmax = vmin, vmax
    else:
        sm_vmin, sm_vmax = None, None
        liv_vmin, liv_vmax = None, None

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    im0 = axes[0].imshow(
        h_sm.T,
        origin="lower",
        aspect="auto",
        extent=[eta_range[0], eta_range[1], phi_range[0], phi_range[1]],
        cmap="viridis",
        vmin=sm_vmin,
        vmax=sm_vmax,
    )

    axes[0].set_title(f"Eta-Phi Distribution ({z_label} SM)", fontsize=18)
    axes[0].set_xlabel(r"$\eta$", fontsize=14)
    axes[0].set_ylabel(r"$\phi$", fontsize=14)

    cbar0 = fig.colorbar(im0, ax=axes[0])
    cbar0.set_label("Counts")

    add_2d_stats_box(
        axes[0],
        f"hEtaPhi_{z_label}_SM",
        eta_sm,
        phi_sm,
        weights=None,
    )

    im1 = axes[1].imshow(
        h_liv.T,
        origin="lower",
        aspect="auto",
        extent=[eta_range[0], eta_range[1], phi_range[0], phi_range[1]],
        cmap="viridis",
        vmin=liv_vmin,
        vmax=liv_vmax,
    )

    axes[1].set_title(f"Eta-Phi Distribution ({z_label} LIV)", fontsize=18)
    axes[1].set_xlabel(r"$\eta$", fontsize=14)
    axes[1].set_ylabel(r"$\phi$", fontsize=14)

    cbar1 = fig.colorbar(im1, ax=axes[1])
    cbar1.set_label("Weighted counts")

    add_2d_stats_box(
        axes[1],
        f"hEtaPhi_{z_label}_LIV",
        eta_liv,
        phi_liv,
        weights=liv_weight,
    )

    fig.text(
        0.02, 0.98, rf"$\theta = {theta_tag}$",
        ha="left",
        va="top",
        fontsize=14,
        bbox=dict(facecolor="white", edgecolor="black", boxstyle="square,pad=0.25", alpha=0.9),
    )
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()

    print("Saved:", output_path)


def plot_z1_z2_variable_2d_heatmaps(
    z1_values,
    z2_values,
    liv_mask,
    liv_weight_used,
    output_path,
    bins=60,
    use_same_color_scale=False,
    manual_ranges=None,
    theta_tag=None,
):
    var_order = ["PT", "Eta", "Phi", "Mass", "Pz", "Energy"]

    fig, axes = plt.subplots(len(var_order), 2, figsize=(15, 24))

    for i, var_name in enumerate(var_order):
        z1_sm = z1_values[var_name]
        z2_sm = z2_values[var_name]

        z1_liv = z1_sm[liv_mask]
        z2_liv = z2_sm[liv_mask]

        if manual_ranges is not None and var_name in manual_ranges:
            var_range = manual_ranges[var_name]
        else:
            var_range = get_common_range(z1_sm, z2_sm, z1_liv, z2_liv)

        h_sm, xedges, yedges = np.histogram2d(
            z1_sm,
            z2_sm,
            bins=bins,
            range=[var_range, var_range],
        )

        h_liv, _, _ = np.histogram2d(
            z1_liv,
            z2_liv,
            bins=[xedges, yedges],
            weights=liv_weight_used,
        )

        if use_same_color_scale:
            vmin = min(np.nanmin(h_sm), np.nanmin(h_liv))
            vmax = max(np.nanmax(h_sm), np.nanmax(h_liv))
            sm_vmin, sm_vmax = vmin, vmax
            liv_vmin, liv_vmax = vmin, vmax
        else:
            sm_vmin, sm_vmax = None, None
            liv_vmin, liv_vmax = None, None

        # ==========================
        # SM
        # ==========================
        ax_sm = axes[i, 0]

        im0 = ax_sm.imshow(
            h_sm.T,
            origin="lower",
            aspect="auto",
            extent=[var_range[0], var_range[1], var_range[0], var_range[1]],
            cmap="viridis",
            vmin=sm_vmin,
            vmax=sm_vmax,
        )

        ax_sm.set_title(f"{var_name}: Z1 vs Z2 (SM)", fontsize=14)
        ax_sm.set_xlabel(f"Z1_{var_name}")
        ax_sm.set_ylabel(f"Z2_{var_name}")
        ax_sm.set_xlim(var_range)
        ax_sm.set_ylim(var_range)

        cbar0 = fig.colorbar(im0, ax=ax_sm)
        cbar0.set_label("Counts")

        add_2d_stats_box(
            ax_sm,
            f"hZ1Z2_{var_name}_SM",
            z1_sm,
            z2_sm,
            weights=None,
        )

        # ==========================
        # LIV
        # ==========================
        ax_liv = axes[i, 1]

        im1 = ax_liv.imshow(
            h_liv.T,
            origin="lower",
            aspect="auto",
            extent=[var_range[0], var_range[1], var_range[0], var_range[1]],
            cmap="viridis",
            vmin=liv_vmin,
            vmax=liv_vmax,
        )

        ax_liv.set_title(f"{var_name}: Z1 vs Z2 (LIV)", fontsize=14)
        ax_liv.set_xlabel(f"Z1_{var_name}")
        ax_liv.set_ylabel(f"Z2_{var_name}")
        ax_liv.set_xlim(var_range)
        ax_liv.set_ylim(var_range)

        cbar1 = fig.colorbar(im1, ax=ax_liv)
        cbar1.set_label("Weighted counts")

        add_2d_stats_box(
            ax_liv,
            f"hZ1Z2_{var_name}_LIV",
            z1_liv,
            z2_liv,
            weights=liv_weight_used,
        )
    
    fig.text(
        0.02, 0.98, rf"$\theta = {theta_tag}$",
        ha="left",
        va="top",
        fontsize=14,
        bbox=dict(facecolor="white", edgecolor="black", boxstyle="square,pad=0.25", alpha=0.9),
    )

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()

    print("Saved:", output_path)

def plot_z1_z2_eta_phi_together(
    z1_eta_sm,
    z1_phi_sm,
    z2_eta_sm,
    z2_phi_sm,
    z1_eta_liv,
    z1_phi_liv,
    z2_eta_liv,
    z2_phi_liv,
    liv_weight,
    output_path,
    bins=60,
    theta_tag=None,
):
    eta_range = (-4, 4)
    phi_range = (-np.pi, np.pi)

    h_z1_sm, xedges, yedges = np.histogram2d(
        z1_eta_sm,
        z1_phi_sm,
        bins=bins,
        range=[eta_range, phi_range],
    )

    h_z2_sm, _, _ = np.histogram2d(
        z2_eta_sm,
        z2_phi_sm,
        bins=[xedges, yedges],
    )

    h_z1_liv, _, _ = np.histogram2d(
        z1_eta_liv,
        z1_phi_liv,
        bins=[xedges, yedges],
        weights=liv_weight,
    )

    h_z2_liv, _, _ = np.histogram2d(
        z2_eta_liv,
        z2_phi_liv,
        bins=[xedges, yedges],
        weights=liv_weight,
    )

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    im0 = axes[0, 0].imshow(
        h_z1_sm.T,
        origin="lower",
        aspect="auto",
        extent=[eta_range[0], eta_range[1], phi_range[0], phi_range[1]],
        cmap="viridis",
    )
    axes[0, 0].set_title("Z1_eta vs Z1_phi (SM)", fontsize=16)
    axes[0, 0].set_xlabel(r"$\eta$", fontsize=14)
    axes[0, 0].set_ylabel(r"$\phi$", fontsize=14)
    fig.colorbar(im0, ax=axes[0, 0]).set_label("Counts")
    add_2d_stats_box(axes[0, 0], "hEtaPhi_Z1_SM", z1_eta_sm, z1_phi_sm)

    im1 = axes[0, 1].imshow(
        h_z2_sm.T,
        origin="lower",
        aspect="auto",
        extent=[eta_range[0], eta_range[1], phi_range[0], phi_range[1]],
        cmap="viridis",
    )
    axes[0, 1].set_title("Z2_eta vs Z2_phi (SM)", fontsize=16)
    axes[0, 1].set_xlabel(r"$\eta$", fontsize=14)
    axes[0, 1].set_ylabel(r"$\phi$", fontsize=14)
    fig.colorbar(im1, ax=axes[0, 1]).set_label("Counts")
    add_2d_stats_box(axes[0, 1], "hEtaPhi_Z2_SM", z2_eta_sm, z2_phi_sm)

    im2 = axes[1, 0].imshow(
        h_z1_liv.T,
        origin="lower",
        aspect="auto",
        extent=[eta_range[0], eta_range[1], phi_range[0], phi_range[1]],
        cmap="viridis",
    )
    axes[1, 0].set_title("Z1_eta vs Z1_phi (LIV)", fontsize=16)
    axes[1, 0].set_xlabel(r"$\eta$", fontsize=14)
    axes[1, 0].set_ylabel(r"$\phi$", fontsize=14)
    fig.colorbar(im2, ax=axes[1, 0]).set_label("Weighted counts")
    add_2d_stats_box(
        axes[1, 0],
        "hEtaPhi_Z1_LIV",
        z1_eta_liv,
        z1_phi_liv,
        weights=liv_weight,
    )

    im3 = axes[1, 1].imshow(
        h_z2_liv.T,
        origin="lower",
        aspect="auto",
        extent=[eta_range[0], eta_range[1], phi_range[0], phi_range[1]],
        cmap="viridis",
    )
    axes[1, 1].set_title("Z2_eta vs Z2_phi (LIV)", fontsize=16)
    axes[1, 1].set_xlabel(r"$\eta$", fontsize=14)
    axes[1, 1].set_ylabel(r"$\phi$", fontsize=14)
    fig.colorbar(im3, ax=axes[1, 1]).set_label("Weighted counts")
    add_2d_stats_box(
        axes[1, 1],
        "hEtaPhi_Z2_LIV",
        z2_eta_liv,
        z2_phi_liv,
        weights=liv_weight,
    )

    fig.text(
        0.02, 0.98, rf"$\theta = {theta_tag}$",
        ha="left",
        va="top",
        fontsize=14,
        bbox=dict(facecolor="white", edgecolor="black", boxstyle="square,pad=0.25", alpha=0.9),
    )

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()

    print("Saved:", output_path)

def plot_eta_sum_diff_together(
    eta_sum_sm,
    eta_diff_sm,
    eta_sum_liv,
    eta_diff_liv,
    liv_weight,
    output_path,
    bins=60,
    theta_tag=None,
):
    eta_range = (-8, 8)

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.hist(
        eta_sum_liv,
        bins=bins,
        range=eta_range,
        weights=liv_weight,
        histtype="stepfilled",
        color="lightgray",
        alpha=0.9,
        label=r"LIV: $\eta(Z1)+\eta(Z2)$",
        zorder=1,
    )

    ax.hist(
        eta_diff_liv,
        bins=bins,
        range=eta_range,
        weights=liv_weight,
        histtype="stepfilled",
        color="khaki",
        alpha=0.8,
        label=r"LIV: $\eta(Z1)-\eta(Z2)$",
        zorder=1,
    )

    ax.hist(
        eta_sum_sm,
        bins=bins,
        range=eta_range,
        histtype="step",
        color="blue",
        linewidth=2,
        linestyle="-",
        label=r"SM: $\eta(Z1)+\eta(Z2)$",
        zorder=3,
    )

    ax.hist(
        eta_diff_sm,
        bins=bins,
        range=eta_range,
        histtype="step",
        color="red",
        linewidth=2,
        linestyle="-",
        label=r"SM: $\eta(Z1)-\eta(Z2)$",
        zorder=3,
    )

    ax.set_title("Eta combinations: SM vs LIV", fontsize=16)
    ax.set_xlabel(r"$\eta$ combination", fontsize=14)
    ax.set_ylabel("Counts / Weighted counts", fontsize=14)
    ax.legend(fontsize=11)

    mean_sum_sm, std_sum_sm = weighted_mean_std(eta_sum_sm)
    mean_diff_sm, std_diff_sm = weighted_mean_std(eta_diff_sm)
    mean_sum_liv, std_sum_liv = weighted_mean_std(eta_sum_liv, liv_weight)
    mean_diff_liv, std_diff_liv = weighted_mean_std(eta_diff_liv, liv_weight)

    text = (
        "SM blue: eta(Z1)+eta(Z2)\n"
        f"Mean {mean_sum_sm: .4g}, Std {std_sum_sm: .4g}\n"
        "SM red: eta(Z1)-eta(Z2)\n"
        f"Mean {mean_diff_sm: .4g}, Std {std_diff_sm: .4g}\n"
        "LIV gray fill: eta(Z1)+eta(Z2)\n"
        f"Mean {mean_sum_liv: .4g}, Std {std_sum_liv: .4g}\n"
        "LIV yellow fill: eta(Z1)-eta(Z2)\n"
        f"Mean {mean_diff_liv: .4g}, Std {std_diff_liv: .4g}"
    )

    ax.text(
        0.02,
        0.98,
        text,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8,
        bbox=dict(
            facecolor="white",
            edgecolor="black",
            boxstyle="square,pad=0.35",
            alpha=0.9,
        ),
    )

    fig.text(
        0.02, 0.98, rf"$\theta = {theta_tag}$",
        ha="left",
        va="top",
        fontsize=14,
        bbox=dict(facecolor="white", edgecolor="black", boxstyle="square,pad=0.25", alpha=0.9),
    )

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()

    print("Saved:", output_path)

def plot_eta_phi_sum_together(
    eta_sum_sm,
    phi_sum_sm,
    eta_sum_liv,
    phi_sum_liv,
    liv_weight,
    output_path,
    bins_eta=60,
    bins_phi=60,
    theta_tag=None,
):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # --------------------------
    # eta sum
    # --------------------------
    ax = axes[0]
    eta_range = (-8, 8)

    ax.hist(
        eta_sum_liv,
        bins=bins_eta,
        range=eta_range,
        weights=liv_weight,
        histtype="stepfilled",
        color="lightgray",
        alpha=0.9,
        label="LIV weighted",
        zorder=1,
    )
    ax.hist(
        eta_sum_sm,
        bins=bins_eta,
        range=eta_range,
        histtype="step",
        color="blue",
        linewidth=2,
        label="SM unweighted",
        zorder=3,
    )

    ax.set_title(r"$(\eta(Z1)+\eta(Z2))/2$", fontsize=16)
    ax.set_xlabel(r"$\eta$ avg", fontsize=13)
    ax.set_ylabel("Counts / Weighted counts", fontsize=13)
    ax.legend()

    # --------------------------
    # phi sum
    # --------------------------
    ax = axes[1]
    phi_range = (-np.pi, np.pi)

    ax.hist(
        phi_sum_liv,
        bins=bins_phi,
        range=phi_range,
        weights=liv_weight,
        histtype="stepfilled",
        color="lightgray",
        alpha=0.9,
        label="LIV weighted",
        zorder=1,
    )
    ax.hist(
        phi_sum_sm,
        bins=bins_phi,
        range=phi_range,
        histtype="step",
        color="blue",
        linewidth=2,
        label="SM unweighted",
        zorder=3,
    )

    # ax.set_title(r"$(\phi(Z1)+\pi-\phi(Z2))/2$", fontsize=16)
    ax.set_title(r"$(\phi(Z1)+\phi(Z2))/2$", fontsize=16)
    ax.set_xlabel(r"$\phi$ avg", fontsize=13)
    ax.set_ylabel("Counts / Weighted counts", fontsize=13)
    ax.set_xticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi])
    ax.set_xticklabels([r"$-\pi$", r"$-\pi/2$", r"$0$", r"$\pi/2$", r"$\pi$"])
    ax.legend()

    fig.text(
        0.02, 0.98, rf"$\theta = {theta_tag}$",
        ha="left",
        va="top",
        fontsize=14,
        bbox=dict(facecolor="white", edgecolor="black", boxstyle="square,pad=0.25", alpha=0.9),
    )

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print("Saved:", output_path)


def plot_eta_phi_diff_together(
    eta_sum_sm,
    phi_sum_sm,
    eta_sum_liv,
    phi_sum_liv,
    liv_weight,
    output_path,
    bins_eta=60,
    bins_phi=60,
    theta_tag=None,
):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # --------------------------
    # eta sum
    # --------------------------
    ax = axes[0]
    eta_range = (-8, 8)

    ax.hist(
        eta_sum_liv,
        bins=bins_eta,
        range=eta_range,
        weights=liv_weight,
        histtype="stepfilled",
        color="lightgray",
        alpha=0.9,
        label="LIV weighted",
        zorder=1,
    )
    ax.hist(
        eta_sum_sm,
        bins=bins_eta,
        range=eta_range,
        histtype="step",
        color="blue",
        linewidth=2,
        label="SM unweighted",
        zorder=3,
    )

    ax.set_title(r"$(\eta(Z1)+\eta(Z2))/2$", fontsize=16)
    ax.set_xlabel(r"$\eta$ avg", fontsize=13)
    ax.set_ylabel("Counts / Weighted counts", fontsize=13)
    ax.legend()

    # --------------------------
    # phi sum
    # --------------------------
    ax = axes[1]
    phi_range = (-np.pi, np.pi)

    ax.hist(
        phi_sum_liv,
        bins=bins_phi,
        range=phi_range,
        weights=liv_weight,
        histtype="stepfilled",
        color="lightgray",
        alpha=0.9,
        label="LIV weighted",
        zorder=1,
    )
    ax.hist(
        phi_sum_sm,
        bins=bins_phi,
        range=phi_range,
        histtype="step",
        color="blue",
        linewidth=2,
        label="SM unweighted",
        zorder=3,
    )

    # ax.set_title(r"$(\phi(Z1)+\pi-\phi(Z2))/2$", fontsize=16)
    # ax.set_title(r"$(\phi(Z1)+\phi(Z2))/2$", fontsize=16)
    ax.set_title(r"$(\phi_1 + \phi_2^{\mathrm{shift}})/2$", fontsize=16)
    ax.set_xlabel(r"$\phi$ avg", fontsize=13)
    ax.set_ylabel("Counts / Weighted counts", fontsize=13)
    ax.set_xticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi])
    ax.set_xticklabels([r"$-\pi$", r"$-\pi/2$", r"$0$", r"$\pi/2$", r"$\pi$"])
    ax.legend()

    fig.text(
        0.02, 0.98, rf"$\theta = {theta_tag}$",
        ha="left",
        va="top",
        fontsize=14,
        bbox=dict(facecolor="white", edgecolor="black", boxstyle="square,pad=0.25", alpha=0.9),
    )

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print("Saved:", output_path)

def plot_five_angles_grid(
    angle_data,
    liv_mask,
    liv_weight_used,
    output_path,
    theta_tag=None,
    bins=60,
):
    fig, axes = plt.subplots(2, 5, figsize=(28, 10))

    angle_configs = [
        ("cos_theta1", r"$\cos\theta_1$", (-1, 1)),
        ("cos_theta2", r"$\cos\theta_2$", (-1, 1)),
        ("phi", r"$\phi$", (-np.pi, np.pi)),
        ("phi1", r"$\phi_1$", (-np.pi, np.pi)),
        ("cos_theta_star", r"$\cos\Theta^*$", (-1, 1)),
    ]

    for i, (key, xlabel, xrng) in enumerate(angle_configs):
        ax_main = axes[0, i]
        ax_ratio = axes[1, i]

        sm_vals = angle_data[key]
        liv_vals = sm_vals[liv_mask]

        finite_sm = np.isfinite(sm_vals)
        finite_liv = np.isfinite(liv_vals) & np.isfinite(liv_weight_used)

        sm_plot = sm_vals[finite_sm]
        liv_plot = liv_vals[finite_liv]
        liv_w = liv_weight_used[finite_liv]

        sm_counts, bin_edges = np.histogram(
            sm_plot,
            bins=bins,
            range=xrng,
        )

        liv_counts, _ = np.histogram(
            liv_plot,
            bins=bin_edges,
            weights=liv_w,
        )

        bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

        ax_main.hist(
            liv_plot,
            bins=bin_edges,
            weights=liv_w,
            histtype="stepfilled",
            color="lightgray",
            alpha=0.9,
            label="LIV weighted",
            zorder=1,
        )

        ax_main.hist(
            sm_plot,
            bins=bin_edges,
            histtype="step",
            color="blue",
            linewidth=2,
            label="SM unweighted",
            zorder=3,
        )

        ymax = ax_main.get_ylim()[1]
        ax_main.set_ylim(0, ymax * 1.20)

        ax_main.set_title(xlabel, fontsize=18)
        ax_main.set_xlabel(xlabel, fontsize=13)
        ax_main.set_ylabel("Counts / Weighted counts", fontsize=13)
        ax_main.legend(fontsize=10)
        ax_main.grid(alpha=0.3)

        mean_sm, std_sm = weighted_mean_std(sm_plot)
        mean_liv, std_liv = weighted_mean_std(liv_plot, liv_w)

        text = (
            f"SM: mean={mean_sm:.4g}, std={std_sm:.4g}\n"
            f"LIV: mean={mean_liv:.4g}, std={std_liv:.4g}"
        )

        ax_main.text(
            0.02,
            0.95,
            text,
            transform=ax_main.transAxes,
            ha="left",
            va="top",
            fontsize=9,
            bbox=dict(
                facecolor="white",
                edgecolor="black",
                boxstyle="square,pad=0.3",
                alpha=0.9,
            ),
        )

        ratio = np.full_like(sm_counts, np.nan, dtype=np.float64)
        nonzero = sm_counts > 0
        ratio[nonzero] = liv_counts[nonzero] / sm_counts[nonzero]

        ax_ratio.step(
            bin_centers,
            ratio,
            where="mid",
            color="red",
            linewidth=1.8,
            label="LIV / SM",
        )

        ax_ratio.axhline(1.0, color="black", linestyle="--", linewidth=1)
        ax_ratio.set_title(f"{xlabel} ratio", fontsize=16)
        ax_ratio.set_xlabel(xlabel, fontsize=13)
        ax_ratio.set_ylabel("LIV / SM", fontsize=13)
        ax_ratio.set_xlim(xrng)
        ax_ratio.grid(alpha=0.3)
        ax_ratio.legend(fontsize=10)

    fig.text(
        0.02, 0.995, rf"$\theta = {theta_tag}$",
        ha="left",
        va="top",
        fontsize=16,
        bbox=dict(facecolor="white", edgecolor="black", boxstyle="square,pad=0.25", alpha=0.9),
    )

    plt.tight_layout(rect=[0, 0, 1, 0.975])
    plt.savefig(output_path, dpi=150)
    plt.close()
    print("Saved:", output_path)

def plot_costheta_phi_phi1_2x2(
    cos_theta_star_sm,
    phi_sm,
    phi1_sm,
    cos_theta_star_liv,
    phi_liv,
    phi1_liv,
    liv_weight,
    output_path,
    bins=(60, 60),
    theta_tag=None,
    use_same_color_scale=False,
):
    x_range = (-1, 1)
    y_range = (-np.pi, np.pi)

    h_sm_phi, xedges_phi, yedges_phi = np.histogram2d(
        cos_theta_star_sm,
        phi_sm,
        bins=bins,
        range=[x_range, y_range],
    )
    h_liv_phi, _, _ = np.histogram2d(
        cos_theta_star_liv,
        phi_liv,
        bins=[xedges_phi, yedges_phi],
        weights=liv_weight,
    )

    h_sm_phi1, xedges_phi1, yedges_phi1 = np.histogram2d(
        cos_theta_star_sm,
        phi1_sm,
        bins=bins,
        range=[x_range, y_range],
    )
    h_liv_phi1, _, _ = np.histogram2d(
        cos_theta_star_liv,
        phi1_liv,
        bins=[xedges_phi1, yedges_phi1],
        weights=liv_weight,
    )

    if use_same_color_scale:
        vmin = 0
        vmax = 400
        # vmin = min(
        #     np.nanmin(h_sm_phi),
        #     np.nanmin(h_liv_phi),
        #     np.nanmin(h_sm_phi1),
        #     np.nanmin(h_liv_phi1),
        # )
        # vmax = max(
        #     np.nanmax(h_sm_phi),
        #     np.nanmax(h_liv_phi),
        #     np.nanmax(h_sm_phi1),
        #     np.nanmax(h_liv_phi1),
        # )
    else:
        vmin, vmax = None, None

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    im00 = axes[0, 0].imshow(
        h_sm_phi.T,
        origin="lower",
        aspect="auto",
        extent=[x_range[0], x_range[1], y_range[0], y_range[1]],
        cmap="viridis",
        vmin=vmin,
        vmax=vmax,
    )
    axes[0, 0].set_title(r"$\phi$ vs $\cos\Theta^*$ (SM)", fontsize=16)
    axes[0, 0].set_xlabel(r"$\cos\Theta^*$", fontsize=14)
    axes[0, 0].set_ylabel(r"$\phi$", fontsize=14)
    fig.colorbar(im00, ax=axes[0, 0]).set_label("Counts")
    add_2d_stats_box(axes[0, 0], "SM", cos_theta_star_sm, phi_sm)

    im01 = axes[0, 1].imshow(
        h_liv_phi.T,
        origin="lower",
        aspect="auto",
        extent=[x_range[0], x_range[1], y_range[0], y_range[1]],
        cmap="viridis",
        vmin=vmin,
        vmax=vmax,
    )
    axes[0, 1].set_title(r"$\phi$ vs $\cos\Theta^*$ (LIV)", fontsize=16)
    axes[0, 1].set_xlabel(r"$\cos\Theta^*$", fontsize=14)
    axes[0, 1].set_ylabel(r"$\phi$", fontsize=14)
    fig.colorbar(im01, ax=axes[0, 1]).set_label("Weighted counts")
    add_2d_stats_box(axes[0, 1], "LIV", cos_theta_star_liv, phi_liv, weights=liv_weight)

    im10 = axes[1, 0].imshow(
        h_sm_phi1.T,
        origin="lower",
        aspect="auto",
        extent=[x_range[0], x_range[1], y_range[0], y_range[1]],
        cmap="viridis",
        vmin=vmin,
        vmax=vmax,
    )
    axes[1, 0].set_title(r"$\phi_1$ vs $\cos\Theta^*$ (SM)", fontsize=16)
    axes[1, 0].set_xlabel(r"$\cos\Theta^*$", fontsize=14)
    axes[1, 0].set_ylabel(r"$\phi_1$", fontsize=14)
    fig.colorbar(im10, ax=axes[1, 0]).set_label("Counts")
    add_2d_stats_box(axes[1, 0], "SM", cos_theta_star_sm, phi1_sm)

    im11 = axes[1, 1].imshow(
        h_liv_phi1.T,
        origin="lower",
        aspect="auto",
        extent=[x_range[0], x_range[1], y_range[0], y_range[1]],
        cmap="viridis",
        vmin=vmin,
        vmax=vmax,
    )
    axes[1, 1].set_title(r"$\phi_1$ vs $\cos\Theta^*$ (LIV)", fontsize=16)
    axes[1, 1].set_xlabel(r"$\cos\Theta^*$", fontsize=14)
    axes[1, 1].set_ylabel(r"$\phi_1$", fontsize=14)
    fig.colorbar(im11, ax=axes[1, 1]).set_label("Weighted counts")
    add_2d_stats_box(axes[1, 1], "LIV", cos_theta_star_liv, phi1_liv, weights=liv_weight)

    fig.text(
        0.02, 0.98, rf"$\theta = {theta_tag}$",
        ha="left",
        va="top",
        fontsize=14,
        bbox=dict(facecolor="white", edgecolor="black", boxstyle="square,pad=0.25", alpha=0.9),
    )

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print("Saved:", output_path)

def plot_z1_z2_eta_phi_1d_sm_liv(
    z1_eta_sm,
    z2_eta_sm,
    z1_phi_sm,
    z2_phi_sm,
    z1_eta_liv,
    z2_eta_liv,
    z1_phi_liv,
    z2_phi_liv,
    liv_weight,
    output_path,
    bins=60,
    theta_tag=None,
):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    configs = [
        (axes[0, 0], z1_eta_sm, z1_eta_liv, (-4, 4), "Z1 Eta", r"$\eta$"),
        (axes[0, 1], z2_eta_sm, z2_eta_liv, (-4, 4), "Z2 Eta", r"$\eta$"),
        (axes[1, 0], z1_phi_sm, z1_phi_liv, (-np.pi, np.pi), "Z1 Phi", r"$\phi$"),
        (axes[1, 1], z2_phi_sm, z2_phi_liv, (-np.pi, np.pi), "Z2 Phi", r"$\phi$"),
    ]

    eta_axes = []

    for ax, sm_vals, liv_vals, xrng, title, xlabel in configs:
        ax.hist(
            liv_vals,
            bins=bins,
            range=xrng,
            weights=liv_weight,
            histtype="stepfilled",
            color="lightgray",
            alpha=0.9,
            label="LIV weighted",
            zorder=1,
        )

        ax.hist(
            sm_vals,
            bins=bins,
            range=xrng,
            histtype="step",
            color="blue",
            linewidth=2,
            label="SM unweighted",
            zorder=3,
        )

        ax.set_title(title, fontsize=16)
        ax.set_xlabel(xlabel, fontsize=13)
        ax.set_ylabel("Counts / Weighted counts", fontsize=13)

        if "Phi" in title:
            ax.set_xticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi])
            ax.set_xticklabels([r"$-\pi$", r"$-\pi/2$", r"$0$", r"$\pi/2$", r"$\pi$"])

        if "Eta" in title:
            eta_axes.append(ax)

        mean_sm, std_sm = weighted_mean_std(sm_vals)
        mean_liv, std_liv = weighted_mean_std(liv_vals, liv_weight)

        text = (
            f"SM: mean={mean_sm:.4g}, std={std_sm:.4g}\n"
            f"LIV: mean={mean_liv:.4g}, std={std_liv:.4g}"
        )

        ax.text(
            0.02,
            0.95,
            text,
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=9,
            bbox=dict(
                facecolor="white",
                edgecolor="black",
                boxstyle="square,pad=0.3",
                alpha=0.9,
            ),
        )

        ax.legend(fontsize=10)
        ax.grid(alpha=0.3)

    # make eta plots share the same y-axis limit
    eta_ymax = max(ax.get_ylim()[1] for ax in eta_axes)
    for ax in eta_axes:
        ax.set_ylim(0, eta_ymax * 1.2)

    # keep phi plots individually padded
    for ax in [axes[1, 0], axes[1, 1]]:
        ymax = ax.get_ylim()[1]
        ax.set_ylim(0, ymax * 1.2)

    fig.text(
        0.02, 0.98, rf"$\theta = {theta_tag}$",
        ha="left",
        va="top",
        fontsize=14,
        bbox=dict(facecolor="white", edgecolor="black", boxstyle="square,pad=0.25", alpha=0.9),
    )

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print("Saved:", output_path)



if __name__ == "__main__":
    theta_tag = "1e-4"

    input_file = f"files/Reconstructed_Z1Z2_LHEF_{theta_tag}.root"
    tree_name = "LHEF"

    output_dir = f"plots/plots_{theta_tag}"
    os.makedirs(output_dir, exist_ok=True)



    remove_negative_liv_weights = False

    bins_2d = 60

    print("Reading:", input_file)

    with uproot.open(input_file) as f:
        tree = f[tree_name]

        z1_pt = tree["Z1_PT"].array(library="np")
        z1_eta = tree["Z1_Eta"].array(library="np")
        z1_phi = tree["Z1_Phi"].array(library="np")
        z1_mass = tree["Z1_Mass"].array(library="np")
        z1_pz = tree["Z1_Pz"].array(library="np")
        z1_energy = tree["Z1_Energy"].array(library="np")

        z2_pt = tree["Z2_PT"].array(library="np")
        z2_eta = tree["Z2_Eta"].array(library="np")
        z2_phi = tree["Z2_Phi"].array(library="np")
        z2_mass = tree["Z2_Mass"].array(library="np")
        z2_pz = tree["Z2_Pz"].array(library="np")
        z2_energy = tree["Z2_Energy"].array(library="np")

        liv_weight = tree["LIV_Weight"].array(library="np")

        cos_theta1 = tree["cos_theta1"].array(library="np")
        cos_theta2 = tree["cos_theta2"].array(library="np")
        phi_cs = tree["phi"].array(library="np")
        phi1_cs = tree["phi1"].array(library="np")
        cos_theta_star = tree["cos_theta_star"].array(library="np")

        
        z_eta_sum = z1_eta + z2_eta
        z_eta_diff = z1_eta - z2_eta
        z_phi_diff = np.mod(z1_phi + np.pi - z2_phi, 2 * np.pi)
        z_phi_sum = np.mod(z1_phi + z2_phi, 2 * np.pi)



    print("Total events:", len(z1_pt))

    if remove_negative_liv_weights:
        liv_mask = liv_weight > 0
    else:
        liv_mask = np.ones_like(liv_weight, dtype=bool)

    liv_weight_used = liv_weight[liv_mask]

    angle_data = {
        "cos_theta1": cos_theta1,
        "cos_theta2": cos_theta2,
        "phi": phi_cs,
        "phi1": phi1_cs,
        "cos_theta_star": cos_theta_star,
    }


    # ==========================
    # simple avg observables
    # ==========================
    eta_sum_avg = (z_eta_sum)/2
    phi_diff_avg = (z_phi_diff)/2
    phi_sum_avg = (z_phi_sum)/2

    phi2_shift = z2_phi - np.pi
    phi2_shift = (phi2_shift + np.pi) % (2 * np.pi) - np.pi
    phi_shift_avg = (z1_phi + phi2_shift) / 2

        
    neg_fraction = np.count_nonzero(liv_weight < 0) / len(liv_weight)
    print(f"Negative weight fraction: {neg_fraction:.4f}")
    print("LIV events used:", np.count_nonzero(liv_mask))
    print("LIV weight sum:", np.sum(liv_weight_used))
    print("Negative LIV weights used:", np.count_nonzero(liv_weight_used < 0))



    plot_z1_z2_eta_phi_together(
        z1_eta_sm=z1_eta,
        z1_phi_sm=z1_phi,
        z2_eta_sm=z2_eta,
        z2_phi_sm=z2_phi,
        z1_eta_liv=z1_eta[liv_mask],
        z1_phi_liv=z1_phi[liv_mask],
        z2_eta_liv=z2_eta[liv_mask],
        z2_phi_liv=z2_phi[liv_mask],
        liv_weight=liv_weight_used,
        output_path=os.path.join(output_dir, "Z1_Z2_eta_phi_SM_LIV_heatmap.png"),
        bins=bins_2d,
        theta_tag=theta_tag,
    )

    plot_eta_sum_diff_together(
        eta_sum_sm=z_eta_sum,
        eta_diff_sm=z_eta_diff,
        eta_sum_liv=z_eta_sum[liv_mask],
        eta_diff_liv=z_eta_diff[liv_mask],
        liv_weight=liv_weight_used,
        output_path=os.path.join(output_dir, "eta_Z1plusZ2_Z1minusZ2_SM_LIV.png"),
        bins=bins_2d,
        theta_tag=theta_tag,
    )

    plot_eta_phi_diff_together(
        eta_sum_sm=eta_sum_avg,
        phi_sum_sm=phi_shift_avg,
        eta_sum_liv=eta_sum_avg[liv_mask],
        phi_sum_liv=phi_shift_avg[liv_mask],
        liv_weight=liv_weight_used,
        output_path=os.path.join(output_dir, "eta_phi_diff_SM_LIV.png"),
        bins_eta=bins_2d,
        bins_phi=bins_2d,
        theta_tag=theta_tag,
    )

    plot_eta_phi_sum_together(
        eta_sum_sm=eta_sum_avg,
        phi_sum_sm=phi_sum_avg,
        eta_sum_liv=eta_sum_avg[liv_mask],
        phi_sum_liv=phi_sum_avg[liv_mask],
        liv_weight=liv_weight_used,
        output_path=os.path.join(output_dir, "eta_phi_sum_SM_LIV.png"),
        bins_eta=bins_2d,
        bins_phi=bins_2d,
        theta_tag=theta_tag,
    )

    # ==========================
    # Eta-Phi heatmaps for Z2
    # left = SM, right = LIV
    # ==========================
    manual_ranges = {
        "PT": (0, 300),
        "Eta": (-4, 4),
        "Phi": (-3.5, 3.5),
        "Mass": (85, 105),
        "Pz": (-300, 300),
        "Energy": (90, 300),
    }

    # ==========================
    # All variable 2D heatmaps:
    # Z1_variable vs Z2_variable
    # left = SM, right = LIV
    # ==========================
    z1_values = {
        "PT": z1_pt,
        "Eta": z1_eta,
        "Phi": z1_phi,
        "Mass": z1_mass,
        "Pz": z1_pz,
        "Energy": z1_energy,
    }

    z2_values = {
        "PT": z2_pt,
        "Eta": z2_eta,
        "Phi": z2_phi,
        "Mass": z2_mass,
        "Pz": z2_pz,
        "Energy": z2_energy,
    }

    plot_z1_z2_variable_2d_heatmaps(
        z1_values=z1_values,
        z2_values=z2_values,
        liv_mask=liv_mask,
        liv_weight_used=liv_weight_used,
        output_path=os.path.join(output_dir, "Z1_vs_Z2_variables_2D_SM_LIV.png"),
        bins=bins_2d,
        use_same_color_scale=True,
        manual_ranges=manual_ranges,
        theta_tag=theta_tag,
    )

    plot_five_angles_grid(
        angle_data=angle_data,
        liv_mask=liv_mask,
        liv_weight_used=liv_weight_used,
        output_path=os.path.join(output_dir, "five_angles_SM_LIV_3x2.png"),
        theta_tag=theta_tag,
        bins=bins_2d,
    )

    plot_costheta_phi_phi1_2x2(
        cos_theta_star_sm=cos_theta_star,
        phi_sm=phi_cs,
        phi1_sm=phi1_cs,
        cos_theta_star_liv=cos_theta_star[liv_mask],
        phi_liv=phi_cs[liv_mask],
        phi1_liv=phi1_cs[liv_mask],
        liv_weight=liv_weight_used,
        output_path=os.path.join(output_dir, "costhetaStar_phi_phi1_2x2_SM_LIV.png"),
        bins=(60, 60),
        theta_tag=theta_tag,
        use_same_color_scale=True,
    )

    plot_z1_z2_eta_phi_1d_sm_liv(
        z1_eta_sm=z1_eta,
        z2_eta_sm=z2_eta,
        z1_phi_sm=z1_phi,
        z2_phi_sm=z2_phi,
        z1_eta_liv=z1_eta[liv_mask],
        z2_eta_liv=z2_eta[liv_mask],
        z1_phi_liv=z1_phi[liv_mask],
        z2_phi_liv=z2_phi[liv_mask],
        liv_weight=liv_weight_used,
        output_path=os.path.join(output_dir, "Z1_Z2_eta_phi_1D_SM_LIV.png"),
        bins=bins_2d,
        theta_tag=theta_tag,
    )



    print("Done.")
