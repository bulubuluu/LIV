"""
Non-commutative Standard Model amplitude calculation with filtered output.
Processes events with specific quark pairs (1&-1 or 2&-2), calculates amplitudes,
and saves only events passing filter criteria to output.
"""

import sys
import numpy as np
import uproot
import awkward as ak
import bsm_nc_module as nc
import os
from typing import Tuple, Optional, Dict, Any

def calculate_amplitudes(
    p1: nc.FourVector, 
    k1: nc.FourVector, 
    k2: nc.FourVector,
    index_position: str = "up"
) -> Tuple[float, float]:
    if index_position not in ["up", "down"]:
        raise ValueError(f"Invalid index position: {index_position}, must be 'up' or 'down'")
    
    try:
        sm_value = nc.sm_amp(p1, k1, k2, index_position)
        nc_value = nc.nc_amp(p1, k1, k2, index_position)
        return sm_value, nc_value
    except Exception as e:
        raise RuntimeError(f"Amplitude calculation failed: {str(e)}")

def identify_particles(
    pids: ak.Array, 
    energies: ak.Array, 
    px: ak.Array, 
    py: ak.Array, 
    pz: ak.Array,
    etas: ak.Array
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
    quark_p4 = None
    anti_quark_p4 = None
    z1_p4 = None
    z2_p4 = None
    
    for i in range(len(pids)):
        current_p4 = np.array([energies[i], px[i], py[i], pz[i]], dtype=np.float64)
        pid = pids[i]
        
        if pid in (1, 2) and quark_p4 is None:
            quark_p4 = current_p4
        elif pid in (-1, -2) and anti_quark_p4 is None:
            anti_quark_p4 = current_p4
        elif pid == 23:
            if z1_p4 is None:
                z1_p4 = current_p4
            elif z2_p4 is None:
                z2_p4 = current_p4
    
    return quark_p4, anti_quark_p4, z1_p4, z2_p4

def filter_events(particle_data: ak.Array) -> Tuple[ak.Array, np.ndarray]:
    pids = particle_data["Particle.PID"]
    etas = particle_data["Particle.Eta"]
    
    has_d_pair = (ak.sum(pids == 1, axis=1) >= 1) & (ak.sum(pids == -1, axis=1) >= 1)
    has_u_pair = (ak.sum(pids == 2, axis=1) >= 1) & (ak.sum(pids == -2, axis=1) >= 1)
    has_two_zs = ak.sum(pids == 23, axis=1) >= 2
    
    event_mask = (has_d_pair | has_u_pair) & has_two_zs
    filtered_data = particle_data[event_mask]
    event_mask_np = ak.to_numpy(event_mask)
    
    print(f"Event filtering complete: Total {len(pids)}, passing {len(filtered_data)}")
    print(f"d-quark pairs: {ak.sum(has_d_pair)}, u-quark pairs: {ak.sum(has_u_pair)}")
    print(f"Events with at least two Z bosons: {ak.sum(has_two_zs)}")
    
    return filtered_data, event_mask_np

def validate_input_file(input_file: str) -> None:
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Input file not found: {input_file}")
    if not os.path.isfile(input_file):
        raise IsADirectoryError(f"Input path is not a file: {input_file}")
    if not os.access(input_file, os.R_OK):
        raise PermissionError(f"No read permission: {input_file}")

def validate_output_directory(output_file: str) -> None:
    output_dir = os.path.dirname(output_file)
    if output_dir and not os.path.exists(output_dir):
        raise FileNotFoundError(f"Output directory not found: {output_dir}")
    if output_dir and not os.access(output_dir, os.W_OK):
        raise PermissionError(f"No write permission: {output_dir}")

def process_file(input_file: str, tree_name: str, output_file: str) -> bool:
    try:
        validate_input_file(input_file)
        validate_output_directory(output_file)
        
        with uproot.open(input_file) as src_file:
            if tree_name not in src_file:
                raise ValueError(f"Tree not found: {tree_name}")
            
            src_tree = src_file[tree_name]
            n_total_events = src_tree.num_entries
            print(f"Reading {n_total_events} events from {input_file}...")
            
            particle_data = src_tree.arrays([
                "Particle.PID", "Particle.Px", "Particle.Py", 
                "Particle.Pz", "Particle.E", "Particle.PT", "Particle.Eta",
                "Particle.Phi", "Particle.M", "Particle.Status", "Particle_size"
            ], library="ak")

            filtered_data, _ = filter_events(particle_data)
            n_filtered_events = len(filtered_data)
            
            if n_filtered_events == 0:
                print("Warning: No qualifying events found")
                return False

            sm_amplitudes = np.full(n_filtered_events, np.nan, dtype=np.float64)
            nc_amplitudes = np.full(n_filtered_events, np.nan, dtype=np.float64)
            
            valid_events = 0
            for event_idx in range(n_filtered_events):
                if event_idx % 1000 == 0 and event_idx > 0:
                    print(f"Processed {event_idx}/{n_filtered_events} events ({valid_events} valid)")
                
                pids = filtered_data["Particle.PID"][event_idx]
                px = filtered_data["Particle.Px"][event_idx]
                py = filtered_data["Particle.Py"][event_idx]
                pz = filtered_data["Particle.Pz"][event_idx]
                energy = filtered_data["Particle.E"][event_idx]
                etas = filtered_data["Particle.Eta"][event_idx]
                
                quark, anti_quark, z1, z2 = identify_particles(
                    pids, energy, px, py, pz, etas
                )
                
                if all(p is not None for p in [quark, anti_quark, z1, z2]):
                    p1 = quark if quark[0] > anti_quark[0] else anti_quark
                    k1, k2 = z1, z2
                    
                    try:
                        sm_amp, nc_amp = calculate_amplitudes(p1, k1, k2, "up")
                        sm_amplitudes[event_idx] = sm_amp
                        nc_amplitudes[event_idx] = nc_amp
                        valid_events += 1
                    except Exception as e:
                        print(f"Event {event_idx} calculation failed: {e}")
            
            print(f"Processing complete. Filtered: {n_filtered_events}, successful calculations: {valid_events}")
            
            with uproot.recreate(output_file) as out_file:
                output_dict: Dict[str, Any] = {
                    "Particle.PID": filtered_data["Particle.PID"],
                    "Particle.Px": filtered_data["Particle.Px"],
                    "Particle.Py": filtered_data["Particle.Py"],
                    "Particle.Pz": filtered_data["Particle.Pz"],
                    "Particle.E": filtered_data["Particle.E"],
                    "Particle.PT": filtered_data["Particle.PT"],
                    "Particle.Eta": filtered_data["Particle.Eta"],
                    "Particle.Phi": filtered_data["Particle.Phi"],
                    "Particle.M": filtered_data["Particle.M"],
                    "Particle.Status": filtered_data["Particle.Status"],
                    "Particle_size": filtered_data["Particle_size"],
                    "SM_Amplitude": sm_amplitudes,
                    "NC_Amplitude": nc_amplitudes
                }
                
                out_file["LHEF"] = output_dict
                
        print(f"Filtered events saved to {os.path.abspath(output_file)}")
        return True
        
    except Exception as e:
        print(f"Error processing {input_file}: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    if len(sys.argv) != 9:
        print("Usage: python3 bsm_nc_calculator.py <input_file> <tree_name> <output_file> <theta_tx> <theta_ty> <theta_xy> <theta_xz> <theta_yz>")
        sys.exit(1)

    input_file = sys.argv[1]
    tree_name = sys.argv[2]
    output_file = sys.argv[3]

    success = process_file(input_file, tree_name, output_file)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()