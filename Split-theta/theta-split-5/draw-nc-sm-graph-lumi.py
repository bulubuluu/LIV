"""
This script reads event data from ROOT files, calculates weights based on SM/NC amplitudes,
generates 1D/2D plots for Z boson kinematic analysis, and exports extreme event data to PDF.
This script reads a ROOT file containing LHEF data, extracts Z boson variables, and plots comparisons between Standard Model and non-commutative amplitudes.
It calculates the weights based on the ratio of non-commutative to Standard Model amplitudes and generates histograms for Z boson properties.
It requires `uproot`, `awkward`, and `matplotlib` libraries for data handling and visualization.
The output plots are saved in a `./plots` directory.   
It is designed to be run from the command line with specified input and output files.

"""


import uproot
import awkward as ak
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.units import cm

def build_theta_tag(theta_tx, theta_ty, theta_xy, theta_xz, theta_yz):
    return f"tx_{theta_tx}_ty_{theta_ty}_xy_{theta_xy}_xz_{theta_xz}_yz_{theta_yz}"

def format_theta_annotation(theta_tx, theta_ty, theta_xy, theta_xz, theta_yz):
    return (
        f"THETA_TX={theta_tx}   THETA_TY={theta_ty}   THETA_XY={theta_xy}\n"
        f"THETA_XZ={theta_xz}   THETA_YZ={theta_yz}"
    )

def add_theta_annotation(ax, theta_text):
    ax.text(
        0.02, 0.98,
        theta_text,
        transform=ax.transAxes,
        ha='left',
        va='top',
        fontsize=9,
        family='monospace',
        bbox=dict(
            boxstyle='round,pad=0.25',
            facecolor='white',
            edgecolor='gray',
            alpha=0.85
        ),
        zorder=10
    )

def add_lumi_annotation(ax, lumi_fb):
    ax.text(
        1.0, 1.04,
        f"Normalized to {lumi_fb:g} fb$^{{-1}}$",
        transform=ax.transAxes,
        ha='right',
        va='bottom',
        fontsize=10,
        clip_on=False
    )

def get_auto_range(data, percentile_low=0.1, percentile_high=99.9, show_all=False):
    """Original 1D plot range logic (unchanged)"""
    if len(data) == 0 or not np.any(np.isfinite(data)):
        return (0, 100)
    
    finite_data = data[np.isfinite(data)]
    if show_all:
        min_val = np.min(finite_data)
        max_val = np.max(finite_data)
    else:
        min_val = np.percentile(finite_data, percentile_low)
        max_val = np.percentile(finite_data, percentile_high)
        if np.isclose(min_val, max_val):
            min_val -= 0.1 * abs(min_val) if min_val != 0 else -1
            max_val += 0.1 * abs(max_val) if max_val != 0 else 1
    return (min_val, max_val)


def plot_2d_distribution(x_data, y_data, weights, title, xlabel, ylabel, filename,
                         theta_text, bins=(50, 50), range=None, cmap='viridis',
                         show_lumi=False, lumi_fb=None):


    """2D plot with tight axis ranges """
    fig, ax = plt.subplots(figsize=(8, 6))
    
    if range is not None:
        ax.set_xlim(range[0])
        ax.set_ylim(range[1])
    
    finite_mask = np.isfinite(x_data) & np.isfinite(y_data)
    if weights is not None:
        finite_mask &= np.isfinite(weights)
        plot_weights = weights[finite_mask]
    else:
        plot_weights = None
    
    x_finite = x_data[finite_mask]
    y_finite = y_data[finite_mask]
    
    hist, xedges, yedges, im = ax.hist2d(
        x_finite, y_finite, bins=bins, range=range, weights=plot_weights, cmap=cmap
    )
    plt.colorbar(im, ax=ax, label='Events')
    
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.grid(alpha=0.3, linestyle='--')
    
    add_theta_annotation(ax, theta_text)
    if show_lumi and lumi_fb is not None:
        add_lumi_annotation(ax, lumi_fb)

    fig.tight_layout()
    fig.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close(fig)
    return fig


def plot_weighted_comparison(original_values, weighted_values, weights, title, xlabel, filename,
                             theta_text, bins=100, range=None, label1="SM", label2="LIV", 
                             color1='blue', color2='red', normalize=False, original_weights=None,
                             values3=None, weights3=None, label3=None, color3='green',
                             show_lumi=False, lumi_fb=None):

    fig, ax = plt.subplots(figsize=(8, 6))

    if range is None:
        combined = []
        if len(original_values) > 0:
            combined.append(original_values[np.isfinite(original_values)])
        if len(weighted_values) > 0:
            combined.append(weighted_values[np.isfinite(weighted_values)])
        if values3 is not None and len(values3) > 0:
            combined.append(values3[np.isfinite(values3)])
        combined = np.concatenate(combined) if combined else np.array([])
        range = get_auto_range(combined, show_all=False)
    
    orig_mask = np.isfinite(original_values)
    if original_weights is not None:
        orig_mask &= np.isfinite(original_weights)
        orig_weights = original_weights[orig_mask]
    else:
        orig_weights = None

    orig_finite = original_values[orig_mask]

    ax.hist(
        orig_finite,
        bins=bins,
        range=range,
        histtype='step',
        linewidth=2,
        color=color1,
        alpha=0.8,
        label=label1,
        density=normalize,
        weights=orig_weights,
    )

    if weights is not None:
        wgt_mask = np.isfinite(weighted_values) & np.isfinite(weights)
        wgt_finite = weighted_values[wgt_mask]
        wgt_weights = weights[wgt_mask]
    else:
        wgt_mask = np.isfinite(weighted_values)
        wgt_finite = weighted_values[wgt_mask]
        wgt_weights = None

    if wgt_weights is not None:
       ax.hist(wgt_finite, bins=bins, range=range, histtype='step',
               linewidth=2, color=color2, alpha=0.8, label=label2, density=normalize,
               weights=wgt_weights)
    
    if values3 is not None and weights3 is not None and label3 is not None:
       val3_finite = values3[np.isfinite(values3)]
       wgt3_finite = weights3[np.isfinite(weights3)]
       ax.hist(val3_finite, bins=bins, range=range, histtype='step',
               linewidth=2, color=color3, alpha=0.8, label=label3, density=normalize,
               weights=wgt3_finite)

    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel('Events', fontsize=12)
    ax.legend(fontsize=10, loc='upper right')
    ax.grid(alpha=0.3)

    add_theta_annotation(ax, theta_text)
    if show_lumi and lumi_fb is not None:
        add_lumi_annotation(ax, lumi_fb)
    fig.tight_layout()
    fig.savefig(filename, dpi=300)
    plt.close(fig)
    return fig


def plot_amplitude_scatter(sm_values, nc_values, title, filename, theta_text,
                           xlabel="SM Amplitude", ylabel="NC Amplitude",
                           alpha=0.5, s=10, xlim=None, ylim=None):

    fig, ax = plt.subplots(figsize=(8, 6))

    if xlim is None:
        xlim = get_auto_range(sm_values, show_all=False)
    if ylim is None:
        ylim = get_auto_range(nc_values, show_all=False)

    finite_mask = np.isfinite(sm_values) & np.isfinite(nc_values)
    ax.scatter(sm_values[finite_mask], nc_values[finite_mask], alpha=alpha, s=s, color='purple')

    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.grid(alpha=0.3)

    add_theta_annotation(ax, theta_text)
    fig.tight_layout()
    fig.savefig(filename, dpi=300)
    plt.close(fig)
    return fig


def plot_weight_distribution(weights, title, filename, theta_text,
                             bins=100, color='orange', range=None, normalize=False,
                             show_lumi=False, lumi_fb=None):

    fig, ax = plt.subplots(figsize=(8, 6))

    if range is None:
        range = get_auto_range(weights, show_all=False)
    
    valid_weights = weights[np.isfinite(weights)]
    ax.hist(valid_weights, bins=bins, color=color, alpha=0.8, density=normalize, range=range)

    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel("Weight Value (1 + NC/SM)", fontsize=12)
    ax.set_ylabel('Events', fontsize=12)
    ax.grid(alpha=0.3)

    add_theta_annotation(ax, theta_text)
    if show_lumi and lumi_fb is not None:
        add_lumi_annotation(ax, lumi_fb)
    fig.tight_layout()
    fig.savefig(filename, dpi=300)
    plt.close(fig)
    return fig

def plot_nc_over_sm_theta_weighted(
    var_nc_sm_theta,
    sm_weights,
    bsm_weights,
    title,
    xlabel,
    filename,
    theta_text,
    bins=100,
    range=None,
    label1="SM",
    label2="LIV",
    color1="blue",
    color2="red",
    normalize=False,
    show_lumi=False, lumi_fb=None
):

    fig, ax = plt.subplots(figsize=(8, 6))

    mask = np.isfinite(var_nc_sm_theta) & np.isfinite(sm_weights) & np.isfinite(bsm_weights)
    x_val = var_nc_sm_theta[mask]
    sm_w_val = sm_weights[mask]
    bsm_w_val = bsm_weights[mask]

    if len(x_val) == 0:
        print(f"Warning: no finite data for {filename}")
        plt.close(fig)
        return None

    if range is None:
        range = get_auto_range(x_val, show_all=False)

    ax.hist(
        x_val,
        bins=bins,
        range=range,
        histtype='step',
        linewidth=2,
        color=color1,
        alpha=0.8,
        label=label1,
        density=normalize,
        weights=sm_w_val
    )

    ax.hist(
        x_val,
        bins=bins,
        range=range,
        histtype='step',
        linewidth=2,
        color=color2,
        alpha=0.8,
        label=label2,
        density=normalize,
        weights=bsm_w_val
    )

    ax.set_xlim(-1.5e5, 1.5e5)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel('Events', fontsize=12)
    ax.legend(fontsize=10, loc='upper right')
    ax.grid(alpha=0.3)

    add_theta_annotation(ax, theta_text)
    if show_lumi and lumi_fb is not None:
        add_lumi_annotation(ax, lumi_fb)
    fig.tight_layout()
    fig.savefig(filename, dpi=300)
    plt.close(fig)
    return fig

def export_extreme_z_quark_momenta_to_pdf(extreme_events, extreme_weights, extreme_indices, output_dir):
    pdf_filename = os.path.join(output_dir, "extreme_events_z_quark_momenta.pdf")
    doc = SimpleDocTemplate(
        pdf_filename,
        pagesize=A4,
        rightMargin=2*cm,
        leftMargin=2*cm,
        topMargin=2*cm,
        bottomMargin=2*cm
    )
    elements = []
    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=16,
        spaceAfter=20,
        alignment=1,
        textColor=colors.darkblue
    )
    section_style = ParagraphStyle(
        'CustomSection',
        parent=styles['Heading2'],
        fontSize=14,
        spaceAfter=15,
        textColor=colors.darkred
    )
    content_style = ParagraphStyle(
        'CustomContent',
        parent=styles['Normal'],
        fontSize=11,
        spaceAfter=10,
        leading=14
    )

    main_title = Paragraph("Extreme Events (1 + NC/SM > 500) - Z Boson & Quark 4-Momentum", title_style)
    elements.append(main_title)
    elements.append(Spacer(1, 12))

    n_extreme = len(extreme_events)
    stats_text = f"Total extreme events found: {n_extreme}\nWeight range: {np.min(extreme_weights):.2f} ~ {np.max(extreme_weights):.2f}"
    stats = Paragraph(stats_text, content_style)
    elements.append(stats)
    elements.append(Spacer(1, 15))

    z_title = Paragraph("1. Z Boson Data (PID = 23, Status = 2)", section_style)
    elements.append(z_title)
    z_headers = [
        "Event Index", "Z Boson Index", "PID", "Status",
        "E (GeV)", "Px (GeV)", "Py (GeV)", "Pz (GeV)", "Weight (1+NC/SM)"
    ]
    z_table_data = [z_headers]

    quark_title = Paragraph("2. Quark Data (PID = ±1, ±2)", section_style)
    elements.append(quark_title)
    quark_headers = [
        "Event Index", "Quark Index", "PID (Quark Type)", "Status",
        "E (GeV)", "Px (GeV)", "Py (GeV)", "Pz (GeV)", "Weight (1+NC/SM)"
    ]
    quark_table_data = [quark_headers]

    for event_idx, (event, weight, global_idx) in enumerate(zip(extreme_events, extreme_weights, extreme_indices)):
        z_mask = (event["Particle.PID"] == 23) & (event["Particle.Status"] == 2)
        if ak.sum(z_mask) > 0:
            z_particles = ak.zip({
                "PID": event["Particle.PID"][z_mask],
                "Status": event["Particle.Status"][z_mask],
                "E": event["Particle.E"][z_mask],
                "Px": event["Particle.Px"][z_mask],
                "Py": event["Particle.Py"][z_mask],
                "Pz": event["Particle.Pz"][z_mask]
            })
            z_np = ak.to_numpy(ak.fill_none(z_particles, np.nan))
            for z_idx, z in enumerate(z_np):
                z_table_data.append([
                    str(global_idx), str(z_idx),
                    str(int(z["PID"])) if np.isfinite(z["PID"]) else "NaN",
                    str(int(z["Status"])) if np.isfinite(z["Status"]) else "NaN",
                    f"{z['E']:.4f}" if np.isfinite(z["E"]) else "NaN",
                    f"{z['Px']:.4f}" if np.isfinite(z["Px"]) else "NaN",
                    f"{z['Py']:.4f}" if np.isfinite(z["Py"]) else "NaN",
                    f"{z['Pz']:.4f}" if np.isfinite(z["Pz"]) else "NaN",
                    f"{weight:.4f}"
                ])

        quark_pids = {1, -1, 2, -2}
        quark_mask = ak.any([event["Particle.PID"] == pid for pid in quark_pids], axis=0)
        if ak.sum(quark_mask) > 0:
            quark_particles = ak.zip({
                "PID": event["Particle.PID"][quark_mask],
                "Status": event["Particle.Status"][quark_mask],
                "E": event["Particle.E"][quark_mask],
                "Px": event["Particle.Px"][quark_mask],
                "Py": event["Particle.Py"][quark_mask],
                "Pz": event["Particle.Pz"][quark_mask]
            })
            quark_np = ak.to_numpy(ak.fill_none(quark_particles, np.nan))
            for q_idx, q in enumerate(quark_np):
                pid_to_quark = {1: "d", -1: "anti-d", 2: "u", -2: "anti-u"}
                quark_type = pid_to_quark.get(int(q["PID"]), f"PID={int(q['PID'])}") if np.isfinite(q["PID"]) else "NaN"
                quark_table_data.append([
                    str(global_idx), str(q_idx),
                    quark_type,
                    str(int(q["Status"])) if np.isfinite(q["Status"]) else "NaN",
                    f"{q['E']:.4f}" if np.isfinite(q["E"]) else "NaN",
                    f"{q['Px']:.4f}" if np.isfinite(q["Px"]) else "NaN",
                    f"{q['Py']:.4f}" if np.isfinite(q["Py"]) else "NaN",
                    f"{q['Pz']:.4f}" if np.isfinite(q["Pz"]) else "NaN",
                    f"{weight:.4f}"
                ])

    if len(z_table_data) > 1:
        z_table = Table(z_table_data, repeatRows=1)
        z_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.lightblue),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.black),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 10),
            ('FONTSIZE', (0, 1), (-1, -1), 9),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.lightgrey])
        ]))
        elements.append(z_table)
    else:
        elements.append(Paragraph("No Z boson data available for extreme events.", content_style))
    elements.append(Spacer(1, 20))

    if len(quark_table_data) > 1:
        quark_table = Table(quark_table_data, repeatRows=1)
        quark_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.lightgreen),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.black),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 10),
            ('FONTSIZE', (0, 1), (-1, -1), 9),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.lightgrey])
        ]))
        elements.append(quark_table)
    else:
        elements.append(Paragraph("No quark data available for extreme events.", content_style))

    doc.build(elements)
    print(f"Extreme events (Z + Quark) momentum PDF saved to: {pdf_filename}")


def main(input_file, tree_name, theta_tx, theta_ty, theta_xy, theta_xz, theta_yz, sigma_b, sigma_sbi0, output_base='./plots'):
    try:
        theta_tag = build_theta_tag(theta_tx, theta_ty, theta_xy, theta_xz, theta_yz)
        theta_text = format_theta_annotation(theta_tx, theta_ty, theta_xy, theta_xz, theta_yz)
        output_dir = os.path.join(output_base, theta_tag)

        output_dir_1d = os.path.join(output_dir, "1D")
        output_dir_2d = os.path.join(output_dir, "2D")
        os.makedirs(output_dir_1d, exist_ok=True)
        os.makedirs(output_dir_2d, exist_ok=True)


        os.makedirs(output_dir, exist_ok=True)
        print(f"Created output directory: {output_dir}")

        branches = [
            "Particle.PID", "Particle.Status", "Particle.M", "Particle.PT", "Particle.Eta", "Particle.Phi",
            "Particle.E", "Particle.Px", "Particle.Py", "Particle.Pz",
            "SM_Amplitude", "NC_Amplitude"
        ]
        print(f"Reading tree '{tree_name}' from file: {input_file}")
        
        with uproot.open(input_file) as file:
            if tree_name not in file:
                raise KeyError(f"Tree '{tree_name}' not found in {input_file}")
            tree = file[tree_name]
            data = tree.arrays(branches, library="ak")
        print(f"Successfully loaded {len(data)} events")

        sm_values = ak.to_numpy(data["SM_Amplitude"])
        nc_values = ak.to_numpy(data["NC_Amplitude"])
        valid_sm_mask = (sm_values != 0) & np.isfinite(sm_values) & np.isfinite(nc_values)
        print(f"Valid events (SM != 0 and finite): {np.sum(valid_sm_mask)}/{len(sm_values)}")
        
        new_weights = np.full_like(sm_values, np.nan, dtype=float)
        new_weights[valid_sm_mask] = 1.0 + (nc_values[valid_sm_mask] / sm_values[valid_sm_mask])
        print("Weight calculation completed: Weight = 1 + NC/SM")

        # Asimov-normalized weights
        lumi = 500000.0      # 500 fb^-1 = 500000 pb^-1
        lumi_fb = 500.0

        # sigma_b = 0.0104701189
        # sigma_sbi0 = 0.0104739004   # pb

        # --------------------------
        # SM Asimov weights
        # --------------------------
        sm_weights = np.full_like(sm_values, np.nan, dtype=float)
        sm_weights[valid_sm_mask] = 1.0

        n_valid = np.sum(valid_sm_mask)
        if n_valid <= 0:
            raise ValueError("No valid SM events found for normalization.")

        sm_weights[valid_sm_mask] = (
            sm_weights[valid_sm_mask] / np.sum(sm_weights[valid_sm_mask]) * lumi 
            * sigma_b
        )

        # --------------------------
        # LIV Asimov weights
        # --------------------------
        finite_weight_mask = np.isfinite(new_weights)
        weight_sum = np.sum(new_weights[finite_weight_mask])

        if weight_sum <= 0:
            raise ValueError("Sum of valid LIV weights is non-positive, cannot normalize.")

        new_weights[finite_weight_mask] = (
            new_weights[finite_weight_mask] / weight_sum * lumi 
            * sigma_sbi0
        )

        print("SM weights normalized to Asimov yield:")
        print("Sum of SM weights =", np.sum(sm_weights[valid_sm_mask]))

        print("LIV weights normalized to Asimov yield:")
        print("Sum of LIV weights =", np.sum(new_weights[finite_weight_mask]))




        print("1D comparison plot of (1+NC/SM) generated")

        theta_components = [
            float(theta_tx),
            float(theta_ty),
            float(theta_xy),
            float(theta_xz),
            float(theta_yz),
        ]

        nonzero_thetas = [x for x in theta_components if not np.isclose(x, 0.0)]

        if len(nonzero_thetas) != 1:
            raise ValueError(
                f"Expected exactly one nonzero theta component, got {len(nonzero_thetas)}: {theta_components}"
            )

        theta_total = nonzero_thetas[0]

        some_array = np.full_like(sm_values, np.nan, dtype=float)
        some_array[valid_sm_mask] = nc_values[valid_sm_mask] / (sm_values[valid_sm_mask] * theta_total)


        plot_weighted_comparison(
            original_values=sm_values[valid_sm_mask],
            weighted_values=nc_values[valid_sm_mask],
            weights=new_weights[valid_sm_mask],
            original_weights=new_weights[valid_sm_mask],
            values3=1+nc_values[valid_sm_mask]/sm_values[valid_sm_mask],
            weights3=new_weights[valid_sm_mask],
            label3="(1 + NC/SM)-(SM)",
            color3='green',
            title="SM vs NC Amplitude vs (1 + NC/SM) Comparison",
            xlabel="Amplitude Value",
            filename=f"{output_dir_1d}/amplitude_comparison.png",
            theta_text=theta_text,
            bins=100,
            range=(-150, 150),
            label1="SM Amplitude-(SM)",
            label2="NC Amplitude-(SM)",
            color1='blue',
            color2='red',
            normalize=False,
            lumi_fb=lumi_fb,
            show_lumi=True
        )

        plot_weighted_comparison(
            original_values=sm_values[valid_sm_mask],
            weighted_values=sm_values[valid_sm_mask],
            weights=new_weights[valid_sm_mask],
            original_weights=sm_weights[valid_sm_mask],
            title="SM Amplitude",
            xlabel="SM Amplitude",
            filename=f"{output_dir_1d}/sm_amplitude_sm_vs_liv.png",
            theta_text=theta_text,
            bins=100,
            range=(-150, 150),
            label1="SM",
            label2="LIV",
            color1='blue',
            color2='red',
            normalize=False,
            lumi_fb=lumi_fb,
            show_lumi=True
        )

        plot_weighted_comparison(
            original_values=nc_values[valid_sm_mask],
            weighted_values=nc_values[valid_sm_mask],
            weights=new_weights[valid_sm_mask],
            original_weights=sm_weights[valid_sm_mask],
            title="NC Amplitude",
            xlabel="NC Amplitude",
            filename=f"{output_dir_1d}/nc_amplitude_sm_vs_liv.png",
            theta_text=theta_text,
            bins=100,
            range=(-150, 150),
            label1="SM",
            label2="LIV",
            color1='blue',
            color2='red',
            normalize=False,
            lumi_fb=lumi_fb,
            show_lumi=True
        )

        plot_nc_over_sm_theta_weighted(
            var_nc_sm_theta=some_array[valid_sm_mask],
            sm_weights=sm_weights[valid_sm_mask],
            bsm_weights=new_weights[valid_sm_mask],
            title="NC/(SM * Theta)",
            xlabel="NC/(SM*theta)",
            filename=f"{output_dir_1d}/nc_over_sm_theta_sm_vs_liv.png",
            theta_text=theta_text,
            lumi_fb=lumi_fb,
            show_lumi=True
        )

        print("Amplitude comparison plot generated")

        # plot_weight_distribution(
        #     weights=new_weights,
        #     title="Weight Distribution (1 + NC/SM) - Valid Events",
        #     filename=f"{output_dir}/weight_distribution_1+NC_SM.png",
        #     bins=100,
        #     color='darkorange',
        #     range=None,
        #     normalize=False,
        #     theta_text=theta_text
        # )
        # print("Weight distribution plot generated")

        plot_amplitude_scatter(
            sm_values=sm_values[valid_sm_mask],
            nc_values=nc_values[valid_sm_mask],
            title="SM Amplitude vs NC Amplitude",
            filename=f"{output_dir_1d}/sm_vs_nc_amplitude_scatter.png",
            theta_text=theta_text,
            alpha=0.5,
            s=5,
            xlim=None,
            ylim=None
        )
        print("SM-NC amplitude scatter plot generated")

        z_mask = (data["Particle.PID"] == 23) & (data["Particle.Status"] == 2)
        total_z_particles = ak.sum(z_mask)
        print(f"Total Z bosons in all events: {total_z_particles}")
        if total_z_particles == 0:
            raise ValueError("No Z bosons found (PID=23, Status=2)")
        
        z_masses = ak.to_numpy(ak.flatten(data["Particle.M"][z_mask], axis=None))
        z_pt = ak.to_numpy(ak.flatten(data["Particle.PT"][z_mask], axis=None))
        z_eta = ak.to_numpy(ak.flatten(data["Particle.Eta"][z_mask], axis=None))
        z_phi = ak.to_numpy(ak.flatten(data["Particle.Phi"][z_mask], axis=None))
        z_energy = ak.to_numpy(ak.flatten(data["Particle.E"][z_mask], axis=None))
        z_pz = ak.to_numpy(ak.flatten(data["Particle.Pz"][z_mask], axis=None))
        
        z_count_per_event = ak.sum(z_mask, axis=1)
        z_count_np = ak.to_numpy(z_count_per_event)
        sum_z_count = np.sum(z_count_np)
        
        assert sum_z_count == total_z_particles, f"Sum of Z per event ({sum_z_count}) != total Z ({total_z_particles})"
        assert len(z_masses) == total_z_particles, f"Z masses length ({len(z_masses)}) != total Z ({total_z_particles})"
        
        z_sm_weights = np.repeat(sm_weights, z_count_np)
        z_new_weights = np.repeat(new_weights, z_count_np)
        
        assert len(z_masses) == len(z_new_weights), \
            f"Z particles count ({len(z_masses)}) != weights count ({len(z_new_weights)})"
        print(f"Found {len(z_masses)} Z bosons, weight array matched (sum Z per event: {sum_z_count})")

        # ------------------------------
        # 2D Plot Configs 
        # ------------------------------
        plot_2d_configs = [
            # 1. pT vs Eta (figure: pT 0-200, Eta -4 to 4)
            (z_pt, z_eta, "pT (GeV)", "Eta (η)", "pt_vs_eta", ((0, 200), (-4, 4))),
            # 2. pT vs Phi (figure: pT 0-200, Phi -3 to 3)
            (z_pt, z_phi, "pT (GeV)", "Phi (φ)", "pt_vs_phi", ((0, 200), (-3, 3))),
            # 3. pZ vs Eta (figure: pZ -100 to 100, Eta -1 to 1)
            (z_pz, z_eta, "pZ (GeV)", "Eta (η)", "pz_vs_eta", ((-100, 100), (-1, 1))),
            # 4. pZ vs Phi (figure: pZ -200 to 200, Phi -3 to 3)
            (z_pz, z_phi, "pZ (GeV)", "Phi (φ)", "pz_vs_phi", ((-200, 200), (-3, 3))),
            # 5. Eta vs Phi (figure: Eta -3 to 3, Phi -3 to 3)
            (z_eta, z_phi, "Eta (η)", "Phi (φ)", "eta_vs_phi", ((-3, 3), (-3, 3))),
            # 6. pT vs Mass (figure: pT 0-150, Mass 85 to 95)
            (z_pt, z_masses, "pT (GeV)", "Mass (GeV)", "pt_vs_mass", ((0, 150), (85, 95))),
            # 7. Eta vs Mass (figure: Eta -4 to 4, Mass 85 to 95)
            (z_eta, z_masses, "Eta (η)", "Mass (GeV)", "eta_vs_mass", ((-4, 4), (85, 95))),
            # 8. pZ vs pT (figure: pZ -200 to 200, pT 0-100)
            (z_pz, z_pt, "pZ (GeV)", "pT (GeV)", "pz_vs_pt", ((-200, 200), (0, 100))),
            # 9. Energy vs pT (figure: Energy 80-200, pT 0-100)
            (z_energy, z_pt, "Energy (GeV)", "pT (GeV)", "energy_vs_pt", ((80, 200), (0, 100)))
        ]

        # Generate 2D plots 
        for x_data, y_data, xlabel, ylabel, plot_name, tight_range in plot_2d_configs:
            plot_2d_distribution(
                x_data=x_data,
                y_data=y_data,
                weights=z_sm_weights,
                title=f"Z Boson {xlabel} vs {ylabel} (SM)",
                xlabel=xlabel,
                ylabel=ylabel,
                filename=f"{output_dir_2d}/z_{plot_name}_2d_nonweighted.png",
                bins=(50, 50),
                theta_text=theta_text,
                range=tight_range,
                lumi_fb=lumi_fb,
                show_lumi=True
            )
            plot_2d_distribution(
                x_data=x_data,
                y_data=y_data,
                weights=z_new_weights,
                title=f"Z Boson {xlabel} vs {ylabel} (LIV)",
                xlabel=xlabel,
                ylabel=ylabel,
                filename=f"{output_dir_2d}/z_{plot_name}_2d_weighted.png",
                bins=(50, 50),
                theta_text=theta_text,
                range=tight_range,
                lumi_fb=lumi_fb,
                show_lumi=True
            )
        print("All 2D distribution plots (tight ranges, no blank areas) generated")

        # ------------------------------
        # 1D Plots 
        # ------------------------------
        z_vars = [
            ("Mass", z_masses, "Mass (GeV)"),
            ("PT", z_pt, "pT (GeV)"),
            ("Eta", z_eta, "Eta (η)"),
            ("Phi", z_phi, "Phi (φ)"),
            ("Energy", z_energy, "Energy (GeV)"),
            ("Pz", z_pz, "Pz (GeV)")
        ]
        all_z_mask = np.ones_like(z_new_weights, dtype=bool)
        print(f"Total Z bosons: {np.sum(all_z_mask)}")

        for var_name, var_data, xlabel in z_vars:
            plot_weighted_comparison(
                original_values=var_data[all_z_mask],
                weighted_values=var_data[all_z_mask],
                weights=z_new_weights[all_z_mask],
                original_weights=z_sm_weights[all_z_mask],
                title=f"Z Boson {var_name} Comparison (All Events)",
                xlabel=xlabel,
                filename=f"{output_dir_1d}/z_{var_name.lower()}_comparison_all_events.png",
                bins=100,
                range=None,  # Original 1D range logic
                label1="SM",
                label2="LIV",
                color1='darkblue',
                color2='crimson',
                normalize=False,
                theta_text=theta_text,
                lumi_fb=lumi_fb,
                show_lumi=True
            )
        print("All Z boson comparison plots (original 1D settings) generated")

        weight_abs_less50_mask = np.abs(z_new_weights) < 50
        weight_abs_more50_mask = np.abs(z_new_weights) > 50
        print(f"Z bosons with |Weight| < 50: {np.sum(weight_abs_less50_mask)}")
        print(f"Z bosons with |Weight| > 50: {np.sum(weight_abs_more50_mask)}")

        for var_name, var_data, xlabel in z_vars:
            if np.sum(weight_abs_less50_mask) > 0:
                plot_weighted_comparison(
                    original_values=var_data[weight_abs_less50_mask],
                    weighted_values=var_data[weight_abs_less50_mask],
                    weights=z_new_weights[weight_abs_less50_mask],
                    original_weights=z_sm_weights[weight_abs_less50_mask],
                    title=f"Z Boson {var_name} Comparison (|Weight| < 50)",
                    xlabel=xlabel,
                    filename=f"{output_dir_1d}/z_{var_name.lower()}_comparison_abs_weight_less50.png",
                    bins=100,
                    range=None,
                    label1="SM",
                    label2="LIV",
                    color1='darkblue',
                    color2='crimson',
                    normalize=False,
                    theta_text=theta_text,
                    lumi_fb=lumi_fb,
                    show_lumi=True
                )
            if np.sum(weight_abs_more50_mask) > 0:
                plot_weighted_comparison(
                    original_values=var_data[weight_abs_more50_mask],
                    weighted_values=var_data[weight_abs_more50_mask],
                    weights=z_new_weights[weight_abs_more50_mask],
                    original_weights=z_sm_weights[weight_abs_more50_mask],
                    title=f"Z Boson {var_name} Comparison (|Weight| > 50)",
                    xlabel=xlabel,
                    filename=f"{output_dir_1d}/z_{var_name.lower()}_comparison_abs_weight_more50.png",
                    bins=100,
                    range=None,
                    label1="SM",
                    label2="LIV",
                    color1='darkblue',
                    color2='crimson',
                    normalize=False,
                    theta_text=theta_text,
                    lumi_fb=lumi_fb,
                    show_lumi=True
                )
        print("Z boson comparison plots (|Weight| < 50 and |Weight| > 50) generated")

        small_pt_mask = z_pt < 5
        huge_weight_mask = np.abs(z_new_weights) > 100
        small_pt_huge_weight_mask = small_pt_mask & huge_weight_mask
        normal_weight_mask = np.abs(z_new_weights) <= 100
        small_pt_normal_weight_mask = small_pt_mask & normal_weight_mask

        print("\n=== Small pT Event Statistics ===")
        print(f"Total Z bosons: {np.sum(all_z_mask)}")
        print(f"Small pT Z bosons (<5 GeV): {np.sum(small_pt_mask)}")
        print(f"Small pT + huge weights (|weight|>100): {np.sum(small_pt_huge_weight_mask)}")
        print(f"Small pT + normal weights (|weight|<=100): {np.sum(small_pt_normal_weight_mask)}")

        peak_check_vars = [
            ("Phi", z_phi, "Phi (φ)"),
            ("Eta", z_eta, "Eta (η)")
        ]

        for var_name, var_data, xlabel in peak_check_vars:
            if np.sum(small_pt_huge_weight_mask) > 0:
                plot_weighted_comparison(
                    original_values=var_data[small_pt_huge_weight_mask],
                    weighted_values=var_data[small_pt_huge_weight_mask],
                    weights=z_new_weights[small_pt_huge_weight_mask],
                    original_weights=z_sm_weights[small_pt_huge_weight_mask],
                    title=f"Z Boson {var_name} - Small pT (<5 GeV) + |Weight|>100",
                    xlabel=xlabel,
                    filename=f"{output_dir_1d}/z_{var_name.lower()}_small_pt_huge_weights.png",
                    bins=100,
                    range=None,
                    label1="SM (Small pT + |W|>100)",
                    label2="LIV (Small pT + |W|>100)",
                    color1='orange',
                    color2='red',
                    normalize=False,
                    theta_text=theta_text,
                    lumi_fb=lumi_fb,
                    show_lumi=True
                )

            if np.sum(small_pt_normal_weight_mask) > 0:
                plot_weighted_comparison(
                    original_values=var_data[small_pt_normal_weight_mask],
                    weighted_values=var_data[small_pt_normal_weight_mask],
                    weights=z_new_weights[small_pt_normal_weight_mask],
                    original_weights=z_sm_weights[small_pt_normal_weight_mask],
                    title=f"Z Boson {var_name} - Small pT (<5 GeV) + |Weight|<=100",
                    xlabel=xlabel,
                    filename=f"{output_dir_1d}/z_{var_name.lower()}_small_pt_normal_weights.png",
                    bins=100,
                    range=None,
                    label1="SM (Small pT + |W|<=100)",
                    label2="LIV (Small pT + |W|<=100)",
                    color1='blue',
                    color2='red',
                    normalize=False,
                    theta_text=theta_text,
                    lumi_fb=lumi_fb,
                    show_lumi=True
                )

        print("Phi/Eta subset plots for small pT analysis generated")

        extreme_mask = (new_weights > 500) & valid_sm_mask
        extreme_indices = np.where(extreme_mask)[0]
        n_extreme_events = len(extreme_indices)
        
        print(f"\nFound {n_extreme_events} extreme events (1 + NC/SM > 500):")
        if n_extreme_events > 0:
            extreme_events = data[extreme_indices]
            extreme_weights = new_weights[extreme_indices]

            for idx, weight in zip(extreme_indices, extreme_weights):
                print(f"  Event {idx}: Weight = {weight:.4f}")

            plot_weight_distribution(
                weights=extreme_weights,
                title="Weight Distribution of Extreme Events (1+NC/SM > 500)",
                filename=f"{output_dir}/extreme_events_weight_distribution.png",
                bins=50,
                color='red',
                range=None,
                normalize=False,
                theta_text=theta_text,
                lumi_fb=lumi_fb,
                show_lumi=True
            )
            print("Extreme events weight distribution plot generated")

            export_extreme_z_quark_momenta_to_pdf(extreme_events, extreme_weights, extreme_indices, output_dir_1d)
        else:
            print("No extreme events found (1 + NC/SM > 500)")

        print(f"\nAll tasks completed! Plots and PDF saved to: {output_dir}")

    except Exception as e:
        print(f"\nError during processing: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    if len(sys.argv) != 10:
        print("Usage: python3 draw-nc-sm-graph-lumi.py <input_file> <tree_name> <theta_tx> <theta_ty> <theta_xy> <theta_xz> <theta_yz> <sigma_b> <sigma_sbi0>")
        sys.exit(1)

    input_file = sys.argv[1]
    tree_name = sys.argv[2]
    theta_tx = sys.argv[3]
    theta_ty = sys.argv[4]
    theta_xy = sys.argv[5]
    theta_xz = sys.argv[6]
    theta_yz = sys.argv[7]
    sigma_b = float(sys.argv[8])
    sigma_sbi0 = float(sys.argv[9])


    if not os.path.exists(input_file):
        print(f"Error: Input file '{input_file}' does not exist")
        sys.exit(1)

    main(input_file, tree_name, theta_tx, theta_ty, theta_xy, theta_xz, theta_yz, sigma_b, sigma_sbi0)

