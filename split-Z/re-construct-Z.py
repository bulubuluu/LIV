import os
import uproot
import awkward as ak
import numpy as np


def reconstruct_Z_from_leptons(pt_evt, eta_evt, phi_evt, mass_evt, pid_evt, pid1, pid2):
    """
    Reconstruct one Z candidate from the first pid1 and first pid2 lepton.

    Returns:
        tuple: pt, eta, phi, mass, pz, energy
        None: if either lepton is missing
    """

    pt_evt = ak.to_numpy(pt_evt)
    eta_evt = ak.to_numpy(eta_evt)
    phi_evt = ak.to_numpy(phi_evt)
    mass_evt = ak.to_numpy(mass_evt)
    pid_evt = ak.to_numpy(pid_evt)

    mask1 = pid_evt == pid1
    mask2 = pid_evt == pid2

    if np.count_nonzero(mask1) == 0 or np.count_nonzero(mask2) == 0:
        return None

    pt1 = float(pt_evt[mask1][0])
    eta1 = float(eta_evt[mask1][0])
    phi1 = float(phi_evt[mask1][0])
    m1 = float(mass_evt[mask1][0])

    pt2 = float(pt_evt[mask2][0])
    eta2 = float(eta_evt[mask2][0])
    phi2 = float(phi_evt[mask2][0])
    m2 = float(mass_evt[mask2][0])

    px1 = pt1 * np.cos(phi1)
    py1 = pt1 * np.sin(phi1)
    pz1 = pt1 * np.sinh(eta1)
    e1 = np.sqrt(px1**2 + py1**2 + pz1**2 + m1**2)

    px2 = pt2 * np.cos(phi2)
    py2 = pt2 * np.sin(phi2)
    pz2 = pt2 * np.sinh(eta2)
    e2 = np.sqrt(px2**2 + py2**2 + pz2**2 + m2**2)

    px = px1 + px2
    py = py1 + py2
    pz = pz1 + pz2
    energy = e1 + e2

    z_pt = np.sqrt(px**2 + py**2)
    z_phi = np.arctan2(py, px)

    z_mass2 = energy**2 - px**2 - py**2 - pz**2
    z_mass = np.sqrt(max(z_mass2, 0.0))

    if z_pt > 0:
        z_eta = np.arcsinh(pz / z_pt)
    else:
        z_eta = 0.0

    return z_pt, z_eta, z_phi, z_mass, pz, energy

def four_vector_from_pt_eta_phi_m(pt, eta, phi, mass):
    px = pt * np.cos(phi)
    py = pt * np.sin(phi)
    pz = pt * np.sinh(eta)
    energy = np.sqrt(px**2 + py**2 + pz**2 + mass**2)
    return np.array([energy, px, py, pz], dtype=np.float64)


def get_first_lepton_p4(pt_evt, eta_evt, phi_evt, mass_evt, pid_evt, target_pid):
    pt_evt = ak.to_numpy(pt_evt)
    eta_evt = ak.to_numpy(eta_evt)
    phi_evt = ak.to_numpy(phi_evt)
    mass_evt = ak.to_numpy(mass_evt)
    pid_evt = ak.to_numpy(pid_evt)

    mask = pid_evt == target_pid
    if np.count_nonzero(mask) == 0:
        return None

    pt = float(pt_evt[mask][0])
    eta = float(eta_evt[mask][0])
    phi = float(phi_evt[mask][0])
    mass = float(mass_evt[mask][0])

    return four_vector_from_pt_eta_phi_m(pt, eta, phi, mass)


def spatial(p4):
    return np.array([p4[1], p4[2], p4[3]], dtype=np.float64)


def unit(vec):
    norm = np.linalg.norm(vec)
    if norm == 0:
        return vec
    return vec / norm


def boost_to_rest_frame(p4, parent_p4):
    """
    Boost p4 into the rest frame of parent_p4.
    Four-vector convention: [E, px, py, pz]
    """
    E = parent_p4[0]
    p = spatial(parent_p4)

    if np.isclose(E, 0.0):
        return p4.copy()

    beta = p / E
    beta2 = np.dot(beta, beta)

    if beta2 <= 0.0 or beta2 >= 1.0:
        return p4.copy()

    gamma = 1.0 / np.sqrt(1.0 - beta2)
    bp = np.dot(beta, spatial(p4))
    factor = ((gamma - 1.0) * bp / beta2) - gamma * p4[0]

    boosted_space = spatial(p4) + factor * beta
    boosted_energy = gamma * (p4[0] - bp)

    return np.array([boosted_energy, boosted_space[0], boosted_space[1], boosted_space[2]], dtype=np.float64)


def signed_angle_between_planes(n1, n2, ref_axis):
    n1u = unit(n1)
    n2u = unit(n2)
    ref = unit(ref_axis)

    cosang = np.clip(np.dot(n1u, n2u), -1.0, 1.0)
    angle = np.arccos(cosang)

    sign = np.sign(np.dot(ref, np.cross(n1u, n2u)))
    if sign == 0:
        sign = 1.0

    return angle * sign


def compute_five_angles_2e2mu(pt_evt, eta_evt, phi_evt, mass_evt, pid_evt):
    """
    Convention used here:
      Z1 = mu- mu+  (13, -13)
      Z2 = e-  e+   (11, -11)

    Returns:
      cos_theta1, cos_theta2, phi, phi1, cos_theta_star
    or None if the event is incomplete.
    """
    mu_minus = get_first_lepton_p4(pt_evt, eta_evt, phi_evt, mass_evt, pid_evt, 13)
    mu_plus  = get_first_lepton_p4(pt_evt, eta_evt, phi_evt, mass_evt, pid_evt, -13)
    e_minus  = get_first_lepton_p4(pt_evt, eta_evt, phi_evt, mass_evt, pid_evt, 11)
    e_plus   = get_first_lepton_p4(pt_evt, eta_evt, phi_evt, mass_evt, pid_evt, -11)

    if any(x is None for x in [mu_minus, mu_plus, e_minus, e_plus]):
        return None

    Z1 = mu_minus + mu_plus
    Z2 = e_minus + e_plus
    ZZ = Z1 + Z2

    # Incoming beam four-vectors (lab frame, arbitrary energy scale)
    beam1 = np.array([1.0, 0.0, 0.0, 1.0], dtype=np.float64)
    beam2 = np.array([1.0, 0.0, 0.0, -1.0], dtype=np.float64)

    # Boost to ZZ rest frame
    Z1_zz = boost_to_rest_frame(Z1, ZZ)
    Z2_zz = boost_to_rest_frame(Z2, ZZ)
    mu_minus_zz = boost_to_rest_frame(mu_minus, ZZ)
    mu_plus_zz = boost_to_rest_frame(mu_plus, ZZ)
    e_minus_zz = boost_to_rest_frame(e_minus, ZZ)
    e_plus_zz = boost_to_rest_frame(e_plus, ZZ)
    beam1_zz = boost_to_rest_frame(beam1, ZZ)
    beam2_zz = boost_to_rest_frame(beam2, ZZ)

    z1_hat = unit(spatial(Z1_zz))
    z2_hat = unit(spatial(Z2_zz))
    b1_hat = unit(spatial(beam1_zz))
    b2_hat = unit(spatial(beam2_zz))

    # Collins-Soper-like axis in ZZ frame
    cs_z = unit(b1_hat - b2_hat)

    # cos(Theta*)
    cos_theta_star = np.clip(np.dot(z1_hat, cs_z), -1.0, 1.0)

   # Boost negative leptons to each Z rest frame
    mu_minus_z1 = boost_to_rest_frame(mu_minus, Z1)
    e_minus_z2 = boost_to_rest_frame(e_minus, Z2)

    # Build reference axes in the same rest frames
    Z2_in_Z1 = boost_to_rest_frame(Z2, Z1)
    Z1_in_Z2 = boost_to_rest_frame(Z1, Z2)

    axis1 = -unit(spatial(Z2_in_Z1))
    axis2 = -unit(spatial(Z1_in_Z2))

    # cos(theta1), cos(theta2) in consistent frames
    cos_theta1 = np.clip(np.dot(unit(spatial(mu_minus_z1)), axis1), -1.0, 1.0)
    cos_theta2 = np.clip(np.dot(unit(spatial(e_minus_z2)), axis2), -1.0, 1.0)

    # Decay plane normals in ZZ frame
    n1 = np.cross(spatial(mu_minus_zz), spatial(mu_plus_zz))
    n2 = np.cross(spatial(e_minus_zz), spatial(e_plus_zz))

    if np.linalg.norm(n1) == 0 or np.linalg.norm(n2) == 0:
        return None

    # Phi = angle between the two decay planes
    phi = signed_angle_between_planes(n1, n2, z1_hat)

    # Production plane normal: beam axis and Z1 direction
    n_prod = np.cross(cs_z, z1_hat)
    if np.linalg.norm(n_prod) == 0:
        return None

    # Phi1 = angle between production plane and Z1 decay plane
    phi1 = signed_angle_between_planes(n_prod, n1, z1_hat)

    return cos_theta1, cos_theta2, phi, phi1, cos_theta_star



if __name__ == "__main__":

    # ==========================
    # Input / output
    # ==========================
    theta = "1e-4"
    input_file = f"/eos/user/y/yzhang4/LIV/LO/MG_LHE_ppZZto4L_LO_theta_{theta}/result/total.root"
    output_file = f"files/Reconstructed_Z1Z2_LHEF_{theta}.root"
    tree_name = "LHEF"

    require_two_status2_Z = True

    # ==========================
    # Read input ROOT tree
    # ==========================
    print("Reading input ROOT file:")
    print(input_file)

    with uproot.open(input_file) as f:
        tree = f[tree_name]

        pid = tree["Particle.PID"].array()
        status = tree["Particle.Status"].array()

        pt = tree["Particle.PT"].array()
        eta = tree["Particle.Eta"].array()
        phi = tree["Particle.Phi"].array()
        mass = tree["Particle.M"].array()

        sm_amp = tree["SM_Amplitude"].array()
        nc_amp = tree["NC_Amplitude"].array()

    # ==========================
    # Compute LIV event weight
    # ==========================
    liv_weight_all = ak.to_numpy(1.0 + nc_amp / sm_amp)

    # ==========================
    # Optional event preselection:
    # exactly two generator Z bosons with PID=23, status=2
    # ==========================
    if require_two_status2_Z:
        is_Z = (pid == 23) & (status == 2)
        mask_twoZ = ak.to_numpy(ak.num(pt[is_Z]) == 2)
        event_indices = np.where(mask_twoZ)[0]
        print("Events with exactly two status=2 Z:", len(event_indices))
    else:
        event_indices = np.arange(len(pid))
        print("Using all events:", len(event_indices))

    # ==========================
    # Reconstruct Z1 and Z2
    # Z1 = mu+ mu-
    # Z2 = e+ e-
    # ==========================
    z1_list = []
    z2_list = []
    selected_liv_weight = []
    selected_event_index = []

    cos_theta1_list = []
    cos_theta2_list = []
    phi_list = []
    phi1_list = []
    cos_theta_star_list = []


    print("Reconstructing Z1=mu+mu- and Z2=e+e- ...")

    for i in event_indices:
        z1 = reconstruct_Z_from_leptons(
            pt[i], eta[i], phi[i], mass[i], pid[i],
            13, -13
        )

        z2 = reconstruct_Z_from_leptons(
            pt[i], eta[i], phi[i], mass[i], pid[i],
            11, -11
        )

        angles = compute_five_angles_2e2mu(
            pt[i], eta[i], phi[i], mass[i], pid[i]
        )

        if z1 is None or z2 is None or angles is None:
            continue

        cos_theta1, cos_theta2, phi_angle, phi1_angle, cos_theta_star = angles

        z1_list.append(z1)
        z2_list.append(z2)
        selected_liv_weight.append(liv_weight_all[i])
        selected_event_index.append(i)

        cos_theta1_list.append(cos_theta1)
        cos_theta2_list.append(cos_theta2)
        phi_list.append(phi_angle)
        phi1_list.append(phi1_angle)
        cos_theta_star_list.append(cos_theta_star)


    if len(z1_list) == 0:
        raise RuntimeError("No events passed the Z1/Z2 reconstruction.")

    z1_array = np.asarray(z1_list, dtype=np.float64)
    z2_array = np.asarray(z2_list, dtype=np.float64)

    liv_weight = np.asarray(selected_liv_weight, dtype=np.float64)
    sm_weight = np.ones_like(liv_weight, dtype=np.float64)
    event_index = np.asarray(selected_event_index, dtype=np.int64)
    neg_fraction = np.count_nonzero(liv_weight < 0) / len(liv_weight)

    # keep original event-level branches for compatibility with current pipeline
    pid_sel = pid[event_index]
    status_sel = status[event_index]

    pt_sel = pt[event_index]
    eta_sel = eta[event_index]
    phi_sel = phi[event_index]
    mass_sel = mass[event_index]

    sm_amp_sel = sm_amp[event_index]
    nc_amp_sel = nc_amp[event_index]

    
    print(f"Negative weight fraction: {neg_fraction:.4f}")
    print("Selected events after lepton reconstruction:", len(event_index))
    print("Positive LIV weights:", np.count_nonzero(liv_weight > 0))
    print("Negative LIV weights:", np.count_nonzero(liv_weight < 0))

    # ==========================
    # Split arrays
    # ==========================
    z1_pt = z1_array[:, 0]
    z1_eta = z1_array[:, 1]
    z1_phi = z1_array[:, 2]
    z1_mass = z1_array[:, 3]
    z1_pz = z1_array[:, 4]
    z1_energy = z1_array[:, 5]

    z2_pt = z2_array[:, 0]
    z2_eta = z2_array[:, 1]
    z2_phi = z2_array[:, 2]
    z2_mass = z2_array[:, 3]
    z2_pz = z2_array[:, 4]
    z2_energy = z2_array[:, 5]

    cos_theta1_arr = np.asarray(cos_theta1_list, dtype=np.float64)
    cos_theta2_arr = np.asarray(cos_theta2_list, dtype=np.float64)
    phi_arr = np.asarray(phi_list, dtype=np.float64)
    phi1_arr = np.asarray(phi1_list, dtype=np.float64)
    cos_theta_star_arr = np.asarray(cos_theta_star_list, dtype=np.float64)

    # ==========================
    # Save reconstructed ROOT file
    # ==========================
    print("Writing output ROOT file:")
    print(output_file)

    with uproot.recreate(output_file) as f:
        f["LHEF"] = {
            "EventIndex": event_index,

            # keep original branches unchanged
            "Particle.PID": pid_sel,
            "Particle.Status": status_sel,
            "Particle.PT": pt_sel,
            "Particle.Eta": eta_sel,
            "Particle.Phi": phi_sel,
            "Particle.M": mass_sel,
            "SM_Amplitude": sm_amp_sel,
            "NC_Amplitude": nc_amp_sel,

            # add reconstructed Z branches
            "Z1_PT": z1_pt.astype(np.float32),
            "Z1_Eta": z1_eta.astype(np.float32),
            "Z1_Phi": z1_phi.astype(np.float32),
            "Z1_Mass": z1_mass.astype(np.float32),
            "Z1_Pz": z1_pz.astype(np.float32),
            "Z1_Energy": z1_energy.astype(np.float32),

            "Z2_PT": z2_pt.astype(np.float32),
            "Z2_Eta": z2_eta.astype(np.float32),
            "Z2_Phi": z2_phi.astype(np.float32),
            "Z2_Mass": z2_mass.astype(np.float32),
            "Z2_Pz": z2_pz.astype(np.float32),
            "Z2_Energy": z2_energy.astype(np.float32),

            "cos_theta1": cos_theta1_arr.astype(np.float32),
            "cos_theta2": cos_theta2_arr.astype(np.float32),
            "phi": phi_arr.astype(np.float32),
            "phi1": phi1_arr.astype(np.float32),
            "cos_theta_star": cos_theta_star_arr.astype(np.float32),

            # convenient weights
            "SM_Weight": sm_weight.astype(np.float32),
            "LIV_Weight": liv_weight.astype(np.float32),
        }

    print("Done.")
    print("Output saved to:", output_file)
