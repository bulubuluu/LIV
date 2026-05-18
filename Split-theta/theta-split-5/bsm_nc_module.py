"""
Non-commutative (NC) Standard Model (SM) scattering amplitude calculations.

This module implements calculations for scattering amplitudes in the non-commutative
extension of the Standard Model. It provides functions for computing both SM and
BSM (Beyond Standard Model) amplitudes for various scattering processes.

The module uses a non-commutative parameter theta to model space-time non-commutativity
and includes calculations for Z-boson production processes.

Adjust the Theta parameter to 1e-4.
"""

import sys
from dataclasses import dataclass, field
from typing import List, Tuple, Union
import math 
import numpy as np
from sympy import symbols, LeviCivita, Matrix

# Type aliases for better readability
FourVector = np.ndarray  # 4-vector with shape (4,)
Tensor4D = np.ndarray   # 4D tensor with shape (4,4,4,4)

@dataclass
class PhysicsConstants:
    """Physical constants used in the calculations."""
    # Electric charge (in units of e)
    E_CHARGE: float = 1.0
    
    # Z-boson mass (in GeV)
    M_Z: float = 91.1876  # PDG 2022 value
    
    # Quark mass (in GeV)
    M_Q: float = 0.0  # Approximating as massless
    
    # Weinberg angle parameters
    SIN2_THETA_W: float = 0.22305  # sin^2(θ_W), PDG 2022 value
    
    # Vector and axial-vector couplings
    C_V: float = 1/2 - 4/3 * SIN2_THETA_W
    C_A: float = 1/2


import sys
from dataclasses import dataclass, field
import numpy as np

@dataclass
class ModelParameters:
    THETA: np.ndarray = field(init=False)

    def __post_init__(self):
        theta_scale_tx = float(sys.argv[4]) if len(sys.argv) > 4 else 0.0
        theta_scale_ty = float(sys.argv[5]) if len(sys.argv) > 5 else 0.0
        theta_scale_xy = float(sys.argv[6]) if len(sys.argv) > 6 else 0.0
        theta_scale_xz = float(sys.argv[7]) if len(sys.argv) > 7 else 0.0
        theta_scale_yz = float(sys.argv[8]) if len(sys.argv) > 8 else 0.0


        theta_tx = np.array([
            [ 0,  1,  0,  0],
            [-1,  0,  0,  0],
            [ 0,  0,  0,  0],
            [ 0,  0,  0,  0]
        ]) * theta_scale_tx

        theta_ty = np.array([
            [ 0,  0,  1,  0],
            [ 0,  0,  0,  0],
            [-1,  0,  0,  0],
            [ 0,  0,  0,  0]
        ]) * theta_scale_ty

        theta_xy = np.array([
            [ 0,  0,  0,  0],
            [ 0,  0,  1,  0],
            [ 0, -1,  0,  0],
            [ 0,  0,  0,  0]
        ]) * theta_scale_xy

        theta_xz = np.array([
            [ 0,  0,  0,  0],
            [ 0,  0,  0,  1],
            [ 0,  0,  0,  0],
            [ 0, -1,  0,  0]
        ]) * theta_scale_xz

        theta_yz = np.array([
            [ 0,  0,  0,  0],
            [ 0,  0,  0,  0],
            [ 0,  0,  0,  1],
            [ 0,  0, -1,  0]
        ]) * theta_scale_yz

        self.THETA = theta_tx + theta_ty + theta_xy + theta_xz + theta_yz

    
    # Amplitude coefficients
    C1: float = 1.0  # Coefficient for terms 21-26
    C2: float = 1.0  # Coefficient for terms 11-14 and 01

# Initialize physics constants and model parameters
PHYSICS = PhysicsConstants()
MODEL = ModelParameters()

# Minkowski metric tensor (η_μν)
ETA: np.ndarray = np.diag([1, -1, -1, -1])

# Define a small epsilon for numerical stability
EPSILON = 1e-12


##########################################################################
##########################################################################
##########################################################################


def random_momenta() -> List[FourVector]:
    """Generate random 4-momenta for a 2->2 scattering process.
    
    Generates random 4-momenta for incoming quarks (p1, p2) and outgoing Z-bosons (k1, k2)
    that satisfy energy-momentum conservation: p1 + p2 = k1 + k2.
    
    Note: Outgoing momenta may not be exactly on-shell.
    
    Returns:
        List[FourVector]: A list of four 4-momenta [p1, p2, k1, k2] where:
            - p1, p2: incoming quark 4-momenta
            - k1, k2: outgoing Z-boson 4-momenta
            Each 4-momentum is a numpy array with shape (4,) in the form (E, px, py, pz)
    """
    # Generate random 3-momenta for incoming quarks
    p1_3mom = np.random.rand(3)
    p2_3mom = np.random.rand(3)
    
    # Calculate energy components using relativistic energy-momentum relation
    p1_energy = np.sqrt(PHYSICS.M_Q**2 + np.dot(p1_3mom, p1_3mom))
    p2_energy = np.sqrt(PHYSICS.M_Q**2 + np.dot(p2_3mom, p2_3mom))
    
    # Construct 4-momenta for incoming quarks
    p1 = np.concatenate(([p1_energy], p1_3mom))
    p2 = np.concatenate(([p2_energy], p2_3mom))
    
    # Generate random 3-momentum for first outgoing Z-boson
    k1_3mom = np.random.rand(3)
    k1_energy = np.sqrt(PHYSICS.M_Z**2 + np.dot(k1_3mom, k1_3mom))
    k1 = np.concatenate(([k1_energy], k1_3mom))
    
    # Calculate second outgoing Z-boson momentum using conservation
    k2 = p1 + p2 - k1
    
    return [p1, p2, k1, k2]
    
    
##########################################################################


# Mandelstam variables
def mand_s(p1, p2, k1, k2):
    return np.dot(p1 + p2, np.dot(ETA, p1 + p2))
    
def mand_t(p1, p2, k1, k2):
    return np.dot(p1 - k1, np.dot(ETA, p1 - k1))

def mand_u(p1, p2, k1, k2):
    return np.dot(p1 - k2, np.dot(ETA, p1 - k2))
    
    
##########################################################################


# tensor contraction
# contracts tensor1 and tensor2 along indices at positions indices1 and indices2
# e.g. A_{\mu\nu\rho} B^{\nu\rho\sigma} = contract(A, [2,3], B, [1,2]) 
def contract(tensor1, indices1, tensor2, indices2):
    indices1 = list(np.asarray(indices1) - 1)
    indices2 = list(np.asarray(indices2) - 1)
    return np.tensordot(tensor1, tensor2, [indices1, indices2])


##########################################################################


# Levi-Civita symbol
# has lower indices (maybe needs a minus sign)
def eps():
    return np.array([[[[LeviCivita(i0, i1, i2, i3) for i3 in range(4)] for i2 in range(4)] for i1 in range(4)] for i0 in range(4)])


##########################################################################


# raises/lowers 4-vector index
# should be generalized to handle several indices
def index_switch(v):
    # return np.dot(ETA, v) 
    return contract(ETA, [2], v, [1])


##########################################################################


# <p1, p2, p3, p4>
# uses momenta with upper indices
def avg(p1, p2, p3, p4):  
    result =  contract(eps(),  [4], p4, [1])
    result =  contract(result, [3], p3, [1])
    result =  contract(result, [2], p2, [1])
    result =  contract(result, [1], p1, [1])
    return result


##########################################################################


# <theta>
# has lower indices
def theta_avg():
    return contract(MODEL.THETA, [1,2], eps(), [1,2]) 
  
  
##########################################################################  
    
    
# v^mu -> (v theta)^mu 
# uses momentum with upper indices
# generates result with an upper index
def theta_dual(v):
    return contract(index_switch(v), [1], MODEL.THETA, [1])
	
	
##########################################################################	
      
        
# amplitudes squared
# use momenta with upper indices


def term01(p1: FourVector, p2: FourVector, k1: FourVector, k2: FourVector,
           s: float, t: float, u: float) -> complex:
    """Calculate the first term (term01) of the non-commutative amplitude.
    
    This term represents the contribution to the amplitude from the non-commutative
    interaction involving the theta-dual of the total incoming momentum.
    
    Args:
        p1: Four-momentum of first incoming quark
        p2: Four-momentum of second incoming quark
        k1: Four-momentum of first outgoing Z-boson
        k2: Four-momentum of second outgoing Z-boson
        s: Mandelstam variable s = (p1 + p2)^2
        t: Mandelstam variable t = (p1 - k1)^2
        u: Mandelstam variable u = (p1 - k2)^2
    
    Returns:
        complex: The contribution to the amplitude from term01
    """
    # Calculate total incoming momentum
    p_total = p1 + p2
    
    # Calculate the theta-dual of total momentum
    p_total_dual = theta_dual(p_total)
    
    # Calculate the antisymmetric product of momenta
    momentum_product = avg(p1, p2, k1, p_total_dual)
    
    # Calculate the propagator terms
    safe_t = t if abs(t) > EPSILON else np.sign(t) * EPSILON
    safe_u = u if abs(u) > EPSILON else np.sign(u) * EPSILON
    propagator_terms = (1/safe_t - 1/safe_u)
    
    # Combine all factors with the coupling constant
    result = momentum_product * propagator_terms * (16 * MODEL.C2)
    
    return result


def term11(p1: FourVector, p2: FourVector, k1: FourVector, k2: FourVector,
           s: float, t: float, u: float) -> complex:
    """Calculate term11 of the non-commutative amplitude.
    
    This term represents the contribution from theta-dual interactions with
    individual incoming quark momenta.
    
    Args:
        p1: Four-momentum of first incoming quark
        p2: Four-momentum of second incoming quark
        k1: Four-momentum of first outgoing Z-boson
        k2: Four-momentum of second outgoing Z-boson
        s: Mandelstam variable s = (p1 + p2)^2
        t: Mandelstam variable t = (p1 - k1)^2
        u: Mandelstam variable u = (p1 - k2)^2
    
    Returns:
        complex: The contribution to the amplitude from term11
    """
    # Calculate theta-dual interactions with individual incoming momenta
    p1_dual_term = avg(p1, p2, k1, theta_dual(p1))
    p2_dual_term = avg(p1, p2, k1, theta_dual(p2))
    
    # Calculate propagator terms
    safe_t = t if abs(t) > EPSILON else np.sign(t) * EPSILON
    safe_u = u if abs(u) > EPSILON else np.sign(u) * EPSILON
    propagator_terms = (1/safe_u - 1/safe_t)
    
    # Combine all factors with the coupling constant
    result = (p1_dual_term + p2_dual_term) * propagator_terms * (4 * MODEL.C2)
    
    return result


def term12(p1: FourVector, p2: FourVector, k1: FourVector, k2: FourVector,
           s: float, t: float, u: float) -> complex:
    """Calculate term12 of the non-commutative amplitude.
    
    This term represents the contribution from theta-dual interactions with
    outgoing Z-boson momenta.
    
    Args:
        p1: Four-momentum of first incoming quark
        p2: Four-momentum of second incoming quark
        k1: Four-momentum of first outgoing Z-boson
        k2: Four-momentum of second outgoing Z-boson
        s: Mandelstam variable s = (p1 + p2)^2
        t: Mandelstam variable t = (p1 - k1)^2
        u: Mandelstam variable u = (p1 - k2)^2
    
    Returns:
        complex: The contribution to the amplitude from term12
    """
    # Calculate theta-dual interactions with outgoing Z-boson momenta
    k1_dual_term = avg(p1, p2, k1, theta_dual(k1))
    k2_dual_term = avg(p1, p2, k2, theta_dual(k2))
    
    # Calculate propagator terms
    safe_t = t if abs(t) > EPSILON else np.sign(t) * EPSILON
    safe_u = u if abs(u) > EPSILON else np.sign(u) * EPSILON
    propagator_terms = (-1) * (1/safe_u + 1/safe_t)
    
    # Combine all factors with the coupling constant
    result = (k1_dual_term + k2_dual_term) * propagator_terms * (4 * MODEL.C2)
    
    return result


def term13(p1: FourVector, p2: FourVector, k1: FourVector, k2: FourVector,
           s: float, t: float, u: float) -> complex:
    """Calculate term13 of the non-commutative amplitude.
    
    This term represents the contribution from theta-average interactions
    with incoming quark momenta.
    
    Args:
        p1: Four-momentum of first incoming quark
        p2: Four-momentum of second incoming quark
        k1: Four-momentum of first outgoing Z-boson
        k2: Four-momentum of second outgoing Z-boson
        s: Mandelstam variable s = (p1 + p2)^2
        t: Mandelstam variable t = (p1 - k1)^2
        u: Mandelstam variable u = (p1 - k2)^2
    
    Returns:
        complex: The contribution to the amplitude from term13
    """
    # Calculate theta-average interaction with incoming momenta
    theta_avg_term = contract(p1, [1], contract(theta_avg(), [2], p2, [1]), [1])
    
    # Calculate propagator terms
    safe_t = t if abs(t) > EPSILON else np.sign(t) * EPSILON
    safe_u = u if abs(u) > EPSILON else np.sign(u) * EPSILON
    propagator_terms = (1/safe_t + 1/safe_u)
    
    # Combine all factors with the coupling constant and Z-boson mass
    result = theta_avg_term * propagator_terms * (8 * MODEL.C2 * PHYSICS.M_Z**2)
    
    return result


def term14(p1: FourVector, p2: FourVector, k1: FourVector, k2: FourVector,
           s: float, t: float, u: float) -> complex:
    """Calculate term14 of the non-commutative amplitude.
    
    This term represents the contribution from theta-average interactions
    with outgoing Z-boson momenta, including mass-dependent terms.
    
    Args:
        p1: Four-momentum of first incoming quark
        p2: Four-momentum of second incoming quark
        k1: Four-momentum of first outgoing Z-boson
        k2: Four-momentum of second outgoing Z-boson
        s: Mandelstam variable s = (p1 + p2)^2
        t: Mandelstam variable t = (p1 - k1)^2
        u: Mandelstam variable u = (p1 - k2)^2
    
    Returns:
        complex: The contribution to the amplitude from term14
    """
    # Calculate theta-average interactions with outgoing Z-boson momenta
    safe_t = t if abs(t) > EPSILON else np.sign(t) * EPSILON
    safe_u = u if abs(u) > EPSILON else np.sign(u) * EPSILON
    k1_term = contract(p1, [1], contract(theta_avg(), [2], k2, [1]), [1]) * (PHYSICS.M_Z**2 - t)
    k2_term = contract(p1, [1], contract(theta_avg(), [2], k1, [1]), [1]) * (PHYSICS.M_Z**2 - u)
    
    # Calculate propagator terms
    propagator_terms = (-1) * (1/safe_u + 1/safe_t)
    
    # Combine all factors with the coupling constant
    result = (k1_term + k2_term) * propagator_terms * (2 * MODEL.C2)
    
    return result
    

def term21(p1: FourVector, p2: FourVector, k1: FourVector, k2: FourVector,
           s: float, t: float, u: float) -> complex:
    """Calculate term21 of the non-commutative amplitude.
    
    This term represents the contribution from theta-dual interactions with
    outgoing Z-boson momenta, including s-channel propagator.
    
    Args:
        p1: Four-momentum of first incoming quark
        p2: Four-momentum of second incoming quark
        k1: Four-momentum of first outgoing Z-boson
        k2: Four-momentum of second outgoing Z-boson
        s: Mandelstam variable s = (p1 + p2)^2
        t: Mandelstam variable t = (p1 - k1)^2
        u: Mandelstam variable u = (p1 - k2)^2
    
    Returns:
        complex: The contribution to the amplitude from term21
    """
    # Calculate theta-dual interactions with outgoing Z-boson momenta
    k1_dual_term = avg(p1, p2, k1, theta_dual(k1))
    k2_dual_term = avg(p1, p2, k2, theta_dual(k2))
    
    # Calculate propagator terms including s-channel
    safe_t = t if abs(t) > EPSILON else np.sign(t) * EPSILON
    safe_u = u if abs(u) > EPSILON else np.sign(u) * EPSILON
    safe_s = s - PHYSICS.M_Z**2
    safe_s = safe_s if abs(safe_s) > EPSILON else np.sign(safe_s) * EPSILON
    propagator_terms = (1/safe_t + 1/safe_u) / safe_s
    
    # Combine all factors with the coupling constant and Z-boson mass
    result = (k1_dual_term + k2_dual_term) * propagator_terms * (MODEL.C1 * PHYSICS.M_Z**2)
    
    return result


def term22(p1: FourVector, p2: FourVector, k1: FourVector, k2: FourVector,
           s: float, t: float, u: float) -> complex:
    """Calculate term22 of the non-commutative amplitude.
    
    This term represents the contribution from theta-dual interactions with
    incoming quark momenta, including s-channel propagator.
    
    Args:
        p1: Four-momentum of first incoming quark
        p2: Four-momentum of second incoming quark
        k1: Four-momentum of first outgoing Z-boson
        k2: Four-momentum of second outgoing Z-boson
        s: Mandelstam variable s = (p1 + p2)^2
        t: Mandelstam variable t = (p1 - k1)^2
        u: Mandelstam variable u = (p1 - k2)^2
    
    Returns:
        complex: The contribution to the amplitude from term22
    """
    # Calculate theta-dual interactions with incoming quark momenta
    p1_dual_term = avg(p1, p2, k2, theta_dual(p1))
    p2_dual_term = avg(p1, p2, k2, theta_dual(p2))
    
    # Calculate propagator terms including s-channel
    safe_t = t if abs(t) > EPSILON else np.sign(t) * EPSILON
    safe_u = u if abs(u) > EPSILON else np.sign(u) * EPSILON
    safe_s = s - PHYSICS.M_Z**2
    safe_s = safe_s if abs(safe_s) > EPSILON else np.sign(safe_s) * EPSILON
    propagator_terms = (1/safe_u - 1/safe_t) / safe_s
    
    # Combine all factors with the coupling constant and Z-boson mass
    result = (p1_dual_term + p2_dual_term) * propagator_terms * (MODEL.C1 * PHYSICS.M_Z**2)
    
    return result


def term23(p1: FourVector, p2: FourVector, k1: FourVector, k2: FourVector,
           s: float, t: float, u: float) -> complex:
    """Calculate term23 of the non-commutative amplitude.
    
    This term represents the contribution from theta-average interactions
    with incoming quark momenta, including s-channel propagator.
    
    Args:
        p1: Four-momentum of first incoming quark
        p2: Four-momentum of second incoming quark
        k1: Four-momentum of first outgoing Z-boson
        k2: Four-momentum of second outgoing Z-boson
        s: Mandelstam variable s = (p1 + p2)^2
        t: Mandelstam variable t = (p1 - k1)^2
        u: Mandelstam variable u = (p1 - k2)^2
    
    Returns:
        complex: The contribution to the amplitude from term23
    """
    # Calculate theta-average interaction with incoming momenta
    theta_avg_term = contract(p1, [1], contract(theta_avg(), [2], p2, [1]), [1])
    
    # Calculate propagator terms including s-channel
    safe_t = t if abs(t) > EPSILON else np.sign(t) * EPSILON
    safe_u = u if abs(u) > EPSILON else np.sign(u) * EPSILON
    safe_s = s - PHYSICS.M_Z**2
    safe_s = safe_s if abs(safe_s) > EPSILON else np.sign(safe_s) * EPSILON
    propagator_terms = (1/safe_t + 1/safe_u) / safe_s
    
    # Combine all factors with the coupling constant and Z-boson mass
    result = theta_avg_term * propagator_terms * (-2 * MODEL.C1 * PHYSICS.M_Z**4)
    
    return result


def term24(p1: FourVector, p2: FourVector, k1: FourVector, k2: FourVector,
           s: float, t: float, u: float) -> complex:
    """Calculate term24 of the non-commutative amplitude.
    
    This term represents the contribution from theta-average interactions
    with outgoing Z-boson momenta, including mass-dependent terms and s-channel propagator.
    
    Args:
        p1: Four-momentum of first incoming quark
        p2: Four-momentum of second incoming quark
        k1: Four-momentum of first outgoing Z-boson
        k2: Four-momentum of second outgoing Z-boson
        s: Mandelstam variable s = (p1 + p2)^2
        t: Mandelstam variable t = (p1 - k1)^2
        u: Mandelstam variable u = (p1 - k2)^2
    
    Returns:
        complex: The contribution to the amplitude from term24
    """
    # Calculate theta-average interactions with outgoing Z-boson momenta
    safe_t = t if abs(t) > EPSILON else np.sign(t) * EPSILON
    safe_u = u if abs(u) > EPSILON else np.sign(u) * EPSILON
    safe_s = s - PHYSICS.M_Z**2
    safe_s = safe_s if abs(safe_s) > EPSILON else np.sign(safe_s) * EPSILON
    k1_term = (PHYSICS.M_Z**2 - t) * contract(p1, [1], contract(theta_avg(), [2], k2, [1]), [1])
    k2_term = (PHYSICS.M_Z**2 - u) * contract(p1, [1], contract(theta_avg(), [2], k1, [1]), [1])
    
    # Calculate propagator terms including s-channel
    propagator_terms = (1/safe_t + 1/safe_u) / safe_s
    
    # Combine all factors with the coupling constant and Z-boson mass
    result = (k1_term + k2_term) * propagator_terms * (MODEL.C1/2 * PHYSICS.M_Z**2)
    
    return result
    
    
def term25(p1: FourVector, p2: FourVector, k1: FourVector, k2: FourVector,
           s: float, t: float, u: float) -> complex:
    """Calculate term25 of the non-commutative amplitude.
    
    This term represents the contribution from theta-dual interactions with
    total incoming momentum, including s-channel propagator.
    
    Args:
        p1: Four-momentum of first incoming quark
        p2: Four-momentum of second incoming quark
        k1: Four-momentum of first outgoing Z-boson
        k2: Four-momentum of second outgoing Z-boson
        s: Mandelstam variable s = (p1 + p2)^2
        t: Mandelstam variable t = (p1 - k1)^2
        u: Mandelstam variable u = (p1 - k2)^2
    
    Returns:
        complex: The contribution to the amplitude from term25
    """
    # Calculate total incoming momentum and its theta-dual
    p_total = p1 + p2
    p_total_dual = theta_dual(p_total)
    
    # Calculate the antisymmetric product with total momentum
    momentum_product = avg(p1, p2, k1, p_total_dual)
    
    # Calculate propagator terms including s-channel
    safe_t = t if abs(t) > EPSILON else np.sign(t) * EPSILON
    safe_u = u if abs(u) > EPSILON else np.sign(u) * EPSILON
    safe_s = s - PHYSICS.M_Z**2
    safe_s = safe_s if abs(safe_s) > EPSILON else np.sign(safe_s) * EPSILON
    propagator_terms = (1/safe_u - 1/safe_t) / safe_s
    
    # Combine all factors with the coupling constant and Z-boson mass
    result = momentum_product * propagator_terms * (4 * MODEL.C1 * PHYSICS.M_Z**2)
    
    return result


def term26(p1: FourVector, p2: FourVector, k1: FourVector, k2: FourVector,
           s: float, t: float, u: float) -> complex:
    """Calculate term26 of the non-commutative amplitude.
    
    This term represents a special contribution involving both theta-dual
    and index-switched momenta, including s-channel propagator.
    
    Args:
        p1: Four-momentum of first incoming quark
        p2: Four-momentum of second incoming quark
        k1: Four-momentum of first outgoing Z-boson
        k2: Four-momentum of second outgoing Z-boson
        s: Mandelstam variable s = (p1 + p2)^2
        t: Mandelstam variable t = (p1 - k1)^2
        u: Mandelstam variable u = (p1 - k2)^2
    
    Returns:
        complex: The contribution to the amplitude from term26
    """
    # Calculate theta-dual of k1 and index-switched k2
    k1_dual = theta_dual(k1)
    k2_switched = index_switch(k2)
    
    # Calculate the contraction and antisymmetric product
    dual_contraction = contract(k1_dual, [1], k2_switched, [1])
    momentum_product = avg(p1, p2, k1, k2)
    
    # Calculate propagator terms including s-channel
    safe_s = s - PHYSICS.M_Z**2
    safe_s = safe_s if abs(safe_s) > EPSILON else np.sign(safe_s) * EPSILON
    propagator_terms = 1.0 / safe_s
    
    # Combine all factors with the coupling constant and Z-boson mass
    result = dual_contraction * momentum_product * propagator_terms * (-MODEL.C1 / PHYSICS.M_Z**2)
    
    return result


def sm_amp(p1: FourVector, k1: FourVector, k2: FourVector, index_position: str) -> float:
    """Calculate the Standard Model (SM) amplitude squared.
    
    This function computes the squared amplitude for the SM process
    q q̄ → Z Z, where q is a quark and Z is a Z-boson.
    
    Args:
        p1: Four-momentum of incoming quark
        k1: Four-momentum of first outgoing Z-boson
        k2: Four-momentum of second outgoing Z-boson
        index_position: Indicates covariance of momentum data, either "up" or "down"
    
    Returns:
        float: The squared amplitude for the SM process
    
    Raises:
        ValueError: If index_position is not "up" or "down"
    """
    if index_position not in ["up", "down"]:
        raise ValueError('index_position must be either "up" or "down"')
    
    # Calculate second incoming quark momentum using conservation
    p2 = k1 + k2 - p1
    
    # Raise momentum indices if needed
    if index_position == "down":
        p1 = index_switch(p1)
        p2 = index_switch(p2)
        k1 = index_switch(k1)
        k2 = index_switch(k2)
    
    # Calculate Mandelstam variables
    s = mand_s(p1, p2, k1, k2)
    t = mand_t(p1, p2, k1, k2)
    u = mand_u(p1, p2, k1, k2)
    
    # Calculate propagator terms
    safe_t = t if abs(t) > EPSILON else np.sign(t) * EPSILON
    safe_u = u if abs(u) > EPSILON else np.sign(u) * EPSILON
    propagator_terms = (safe_t/safe_u + safe_u/safe_t + 4*s*PHYSICS.M_Z**2/(safe_u*safe_t) - 
                       PHYSICS.M_Z**4*(1/safe_u**2 + 1/safe_t**2))
    
    # Calculate coupling constant terms
    coupling_terms = (2 * PHYSICS.E_CHARGE**4 / 
                     (16 * PHYSICS.SIN2_THETA_W**2 * (1 - PHYSICS.SIN2_THETA_W)**2))
    
    # Calculate vector and axial-vector coupling terms
    v_a_terms = ((PHYSICS.C_V**2 + PHYSICS.C_A**2)**2 + 
                 4 * PHYSICS.C_V**2 * PHYSICS.C_A**2)
    
    # Combine all terms
    result = propagator_terms * coupling_terms * v_a_terms
    
    return result


def nc_amp(p1: FourVector, k1: FourVector, k2: FourVector, index_position: str) -> float:
    """Calculate the non-commutative (NC) amplitude squared.
    
    This function computes the squared amplitude for the NC process
    q q̄ → Z Z, including all non-commutative corrections up to first order
    in the theta parameter.
    
    Args:
        p1: Four-momentum of incoming quark
        k1: Four-momentum of first outgoing Z-boson
        k2: Four-momentum of second outgoing Z-boson
        index_position: Indicates covariance of momentum data, either "up" or "down"
    
    Returns:
        float: The squared amplitude for the NC process, including all terms
    
    Raises:
        ValueError: If index_position is not "up" or "down"
    """
    if index_position not in ["up", "down"]:
        raise ValueError('index_position must be either "up" or "down"')
    
    # Calculate second incoming quark momentum using conservation
    p2 = k1 + k2 - p1
    
    # Raise momentum indices if needed
    if index_position == "down":
        p1 = index_switch(p1)
        p2 = index_switch(p2)
        k1 = index_switch(k1)
        k2 = index_switch(k2)
    
    # Calculate Mandelstam variables
    s = mand_s(p1, p2, k1, k2)
    t = mand_t(p1, p2, k1, k2)
    u = mand_u(p1, p2, k1, k2)
    
    # Calculate all NC terms
    result = 0.0
    result += term01(p1, p2, k1, k2, s, t, u)
    result += term11(p1, p2, k1, k2, s, t, u)
    result += term12(p1, p2, k1, k2, s, t, u)
    result += term13(p1, p2, k1, k2, s, t, u)
    result += term14(p1, p2, k1, k2, s, t, u)
    result += term21(p1, p2, k1, k2, s, t, u)
    result += term22(p1, p2, k1, k2, s, t, u)
    result += term23(p1, p2, k1, k2, s, t, u)
    result += term24(p1, p2, k1, k2, s, t, u)
    result += term25(p1, p2, k1, k2, s, t, u)
    result += term26(p1, p2, k1, k2, s, t, u)
    
    # Apply the overall normalization factor
    # Note: The factor 6.0 is a rescaling factor that needs to be verified
    result /= 6.0
    
    return result

    
##########################################################################


def random_momenta_onshell() -> List[FourVector]:
    """Generate random 4-momenta for a 2->2 scattering process with all momenta on-shell.
    
    Generates random 4-momenta for incoming quarks (p1, p2) and outgoing Z-bosons (k1, k2)
    that satisfy:
      - energy-momentum conservation: p1 + p2 = k1 + k2
      - all momenta are on-shell: p1^2 = m_q^2, p2^2 = m_q^2, k1^2 = m_Z^2, k2^2 = m_Z^2
    
    Returns:
        List[FourVector]: A list of four 4-momenta [p1, p2, k1, k2]
    """
    m1 = PHYSICS.M_Q
    m2 = PHYSICS.M_Q
    m3 = PHYSICS.M_Z
    m4 = PHYSICS.M_Z

    # Choose random total energy above threshold
    E_cm = m3 + m4 + 10.0  # 10 GeV above threshold
    # Randomize direction for k1 in CM frame
    costheta = 2 * np.random.rand() - 1
    sintheta = np.sqrt(1 - costheta**2)
    phi = 2 * np.pi * np.random.rand()

    # 2->2 kinematics in CM frame
    def momentum_mag(E_cm, mA, mB):
        return np.sqrt((E_cm**2 - (mA + mB)**2) * (E_cm**2 - (mA - mB)**2)) / (2 * E_cm)

    p_in = momentum_mag(E_cm, m1, m2)
    p_out = momentum_mag(E_cm, m3, m4)

    # Incoming quarks (along z)
    p1 = np.array([np.sqrt(m1**2 + p_in**2), 0, 0, +p_in])
    p2 = np.array([np.sqrt(m2**2 + p_in**2), 0, 0, -p_in])

    # Outgoing Zs (random direction)
    k1 = np.array([
        np.sqrt(m3**2 + p_out**2),
        p_out * sintheta * np.cos(phi),
        p_out * sintheta * np.sin(phi),
        p_out * costheta
    ])
    k2 = np.array([
        np.sqrt(m4**2 + p_out**2),
        -p_out * sintheta * np.cos(phi),
        -p_out * sintheta * np.sin(phi),
        -p_out * costheta
    ])
    return [p1, p2, k1, k2]

    
##########################################################################



