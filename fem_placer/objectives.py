"""
Objective functions for FPGA placement optimization using QUBO formulation.

This module provides loss functions and inference methods for FPGA placement
using the QUBO (Quadratic Unconstrained Binary Optimization) approach with
site coordinate matrices.

Key functions:
- HPWL loss calculation (get_hpwl_loss_qubo)
- Constraint loss for preventing overlaps (get_constraints_loss)
- Expected placement with free energy minimization (expected_fpga_placement)
- Inference for extracting final placements (infer_placements)
"""

import torch
import torch.nn.functional as Func
from typing import Dict
from .logger import *

# Global history tracking for optimization visualization
_hpwl_loss_history = []
_constrain_loss_history = []
_total_loss_history = []
_placement_history = []

show_steps = [50, 100, 150, 199]


def get_loss_history():
    """Get the loss history from optimization."""
    return {
        'hpwl_losses': _hpwl_loss_history.copy(),
        'constrain_losses': _constrain_loss_history.copy(),
        'total_losses': _total_loss_history.copy()
    }


def get_placement_history():
    """Get the placement history from optimization."""
    return _placement_history.copy()


def clear_history():
    """Clear the loss and placement history."""
    global _hpwl_loss_history, _constrain_loss_history, _total_loss_history, _placement_history
    _hpwl_loss_history = []
    _constrain_loss_history = []
    _total_loss_history = []
    _placement_history = []


# =============================================================================
# Coordinate Functions (QUBO approach)
# =============================================================================

def count_duplicate_coords(coords):
    """
    Count duplicate coordinate pairs and calculate sum of (count - 1) for each unique pair.

    For each unique coordinate, if it appears a_i times, add (a_i - 1) to the total.
    This measures the total number of "extra" duplicates.

    Args:
        coords: Tensor of coordinates [num_coords, 2] or [batch_size, num_coords, 2]

    Returns:
        duplicate_count: Sum of (a_i - 1) for all unique coordinate sets
    """
    # Handle batch dimension if present
    if coords.dim() == 3:
        coords = coords[0]  # Take first batch element
    
    # Convert to tuple format for comparison
    # Stack into a single tensor for unique operation
    coords_int = coords.long()
    
    # Use torch.unique with return_counts
    unique_coords, counts = torch.unique(coords_int, dim=0, return_counts=True)
    
    # Calculate sum(a_i - 1) where a_i is the count for each unique coordinate
    duplicate_sum = torch.sum(counts - 1).item()
    
    return duplicate_sum, unique_coords, counts


def get_site_distance_matrix(coords):
    """
    Calculate Manhattan distance matrix between all pairs of sites.

    Args:
        coords: Site coordinates [num_sites, 2]

    Returns:
        distances: Distance matrix [num_sites, num_sites]
    """
    coords_i = coords.unsqueeze(1)
    coords_j = coords.unsqueeze(0)
    distances = torch.sum(torch.abs(coords_i - coords_j), dim=2)
    return distances


def get_expected_placements_from_index(p, site_coords_matrix):
    """
    Get expected placement coordinates from probability distribution.

    Args:
        p: Probability distribution [batch_size, num_instances, num_sites]
        site_coords_matrix: Site coordinates [num_sites, 2]

    Returns:
        expected_coords: Expected coordinates [batch_size, num_instances, 2]
    """
    expected_coords = torch.matmul(p, site_coords_matrix)
    return expected_coords


def get_hard_placements_from_index(p, site_coords_matrix):
    """
    Get hard (discrete) placement coordinates using argmax.

    Args:
        p: Probability distribution [batch_size, num_instances, num_sites]
        site_coords_matrix: Site coordinates [num_sites, 2]

    Returns:
        hard_coords: Hard placement coordinates [batch_size, num_instances, 2]
    """
    site_indices = torch.argmax(p, dim=2)
    hard_coords = site_coords_matrix[site_indices]
    return hard_coords


def get_placements_from_index_st(p, site_coords_matrix):
    """
    Get placements using straight-through estimator.

    Args:
        p: Probability distribution [batch_size, num_instances, num_sites]
        site_coords_matrix: Site coordinates [num_sites, 2]

    Returns:
        straight_coords: Coordinates with straight-through gradient [batch_size, num_instances, 2]
    """
    with torch.no_grad():
        site_indices = torch.argmax(p, dim=2)
        hard_coords = site_coords_matrix[site_indices]

    expected_coords = torch.matmul(p, site_coords_matrix)
    straight_coords = expected_coords + (hard_coords - expected_coords).detach()

    return straight_coords


# =============================================================================
# HPWL Loss Functions (QUBO approach)
# =============================================================================

def get_hpwl_loss_qubo(J, p, D):
    """
    Calculate HPWL loss using QUBO formulation.

    Args:
        J: Coupling matrix [num_instances, num_instances]
        p: Probability distribution [batch_size, num_instances, num_sites]
        site_coords_matrix: Site coordinates [num_sites, 2]

    Returns:
        total_wirelength: Total wirelength for each batch [batch_size]
    """
    # Batch matrix multiplication: (p @ D) @ p^T
    PD = torch.matmul(p, D)
    P_transposed = p.transpose(1, 2)
    E_matrix = torch.bmm(PD, P_transposed)

    return 0.5 * torch.sum(E_matrix * J.unsqueeze(0), dim=(1, 2))


def get_hpwl_loss_qubo_sparse_accel(J, p, D):
    """
    Calculate HPWL loss using block-sparse matrix multiplication acceleration.
    """
    
    batch_size, N, M = p.shape
    device = p.device
    
    indices = J.indices()
    values = J.values()
    
    # 预构建块对角稀疏矩阵 indices
    # shape: (2, batch_size * nnz)
    offsets = torch.arange(batch_size, device=device) * N
    # indices[0] shape: (nnz), offsets shape: (batch_size)
    # broadcasting: (batch_size, nnz) -> flatten
    
    rows = (indices[0].unsqueeze(0) + offsets.unsqueeze(1)).flatten()
    cols = (indices[1].unsqueeze(0) + offsets.unsqueeze(1)).flatten()
    vals = values.repeat(batch_size)
    
    block_indices = torch.stack([rows, cols])
    block_shape = (batch_size * N, batch_size * N)
    
    # 构建块对角稀疏矩阵 B_J
    # [batch*N, batch*N]
    B_J = torch.sparse_coo_tensor(block_indices, vals, block_shape, device=device)
    
    # reshape p to [batch*N, M]
    p_flat = p.reshape(batch_size * N, M)
    
    # 1. Sparse MM: B_p = B_J @ p_flat ( [batch*N, batch*N] @ [batch*N, M] -> [batch*N, M] )
    B_p = torch.sparse.mm(B_J, p_flat)
    
    # Reshape back to batch: [batch, N, M]
    B_p_batch = B_p.reshape(batch_size, N, M)
    
    # 2. Dense BMM: C = p^T @ B_p_batch ( [batch, M, N] @ [batch, N, M] -> [batch, M, M] )
    p_T = p.transpose(1, 2)
    C = torch.bmm(p_T, B_p_batch)
    
    # 3. Element-wise mul & sum: sum( D * C )
    loss = 0.5 * torch.sum(D * C, dim=(1, 2))
    
    return loss


def get_hpwl_loss_qubo_with_io(J_LL, J_LI, 
                               p_logic, p_io,
                               D_LL, D_LI):
    """
    Calculate HPWL loss including IO connections.

    Args:
        J_LL: Logic-logic coupling matrix [num_logic, num_logic]
        J_LI: Logic-IO coupling matrix [num_logic, num_io]
        p_logic: Logic probability distribution [batch_size, num_logic, num_logic_sites]
        p_io: IO probability distribution [batch_size, num_io, num_io_sites]
        logic_site_coords_matrix: Logic site coordinates [num_logic_sites, 2]
        io_site_coords_matrix: IO site coordinates [num_io_sites, 2]

    Returns:
        total_wl: Total wirelength for each batch [batch_size]
    """
    batch_size, n_logic, _ = p_logic.shape
    device = p_logic.device

    total_wl = torch.zeros(batch_size, device=device)

    # Logic-logic wirelength
    PD = torch.matmul(p_logic, D_LL)
    p_logic_T = p_logic.transpose(1, 2)
    E_LL = torch.bmm(PD, p_logic_T)
    wl_LL = 0.5 * torch.sum(E_LL * J_LL.unsqueeze(0) , dim=(1, 2))
    total_wl += wl_LL

    # Logic-IO wirelength
    PD_LI = torch.matmul(p_logic, D_LI)
    p_io_T = p_io.transpose(1, 2)
    E_LI = torch.bmm(PD_LI, p_io_T)
    wl_LI = torch.sum(E_LI * J_LI.unsqueeze(0), dim=(1, 2))
    total_wl += wl_LI

    return total_wl


# =============================================================================
# Constraint Loss Functions (QUBO approach)
# =============================================================================

def get_constraints_loss(p, alpha):
    """
    Calculate site usage constraint loss.

    Args:
        p: Probability distribution [batch_size, num_instances, num_sites]

    Returns:
        site_constraint: Constraint loss for each batch [batch_size]
    """
    site_usage = torch.sum(p, dim=1)
    site_constraint = torch.sum(30 * Func.softplus(site_usage - 1)**2, dim=1)
    return alpha * site_constraint

# def get_constraints_loss_rev(p, alpha):
#     """
#     Calculate site capacity constraint loss following the QUBO formulation in Eq. (10).

#     Args:
#         p: Probability distribution [batch_size, num_instances, num_sites]
#         alpha: Penalty coefficient (λ in Eq. (10))

#     Returns:
#         site_constraint: Constraint loss for each batch [batch_size]
#     """
#     site_usage = torch.sum(p, dim=1)                     # [batch_size, num_sites]
#     # Standard quadratic penalty: (∑ p_ij - 1)^2
#     site_constraint = 30 * torch.sum((site_usage - 1) ** 2, dim=1)
#     return alpha * site_constraint


def get_constraints_loss_with_io(p_logic, p_io, alpha, beta):
    """
    Calculate constraint loss for both logic and IO placements.

    Args:
        p_logic: Logic probability distribution [batch_size, num_logic, num_logic_sites]
        p_io: IO probability distribution [batch_size, num_io, num_io_sites]

    Returns:
        constraint_loss: Total constraint loss [batch_size]
    """
    coeff_1 = p_logic.shape[1] / 2
    logic_site_usage = torch.sum(p_logic, dim=1)
    logic_constraint = torch.sum(coeff_1 * Func.softplus(logic_site_usage - 1)**2, dim=1)

    coeff_2 = p_io.shape[1] / 20
    io_site_usage = torch.sum(p_io, dim=1)
    io_constraint = torch.sum(coeff_2 * Func.softplus(io_site_usage - 1)**2, dim=1)
    return alpha * logic_constraint + beta * io_constraint

# 

# =============================================================================
# N-Region Generalised Objective (CLB + IO + DSP + …)
# =============================================================================

def get_hpwl_loss_qubo_with_regions(
    region_ps: Dict[str, torch.Tensor],
    region_coupling: Dict[str, Dict[str, torch.Tensor]],
    region_distances: Dict[str, Dict[str, torch.Tensor]],
) -> torch.Tensor:
    """
    Generalised HPWL loss for N regions.

    ``region_ps`` maps region name → probability tensor ``[B, N_i, M_i]``.
    ``region_coupling`` maps ``(rA, rB)`` → coupling matrix ``[N_A, N_B]``.
    ``region_distances`` maps ``(rA, rB)`` → distance matrix ``[M_A, M_B]``.

    Returns total wirelength ``[B]``.
    """
    total_wl = 0
    for rA in region_ps:
        pA = region_ps[rA]
        for rB in region_ps:
            J = region_coupling.get(rA, {}).get(rB)
            D = region_distances.get(rA, {}).get(rB)
            if J is None or D is None or J.numel() == 0:
                continue
            pB = region_ps[rB]
            PD = torch.matmul(pA, D)
            pBT = pB.transpose(1, 2)
            E = torch.bmm(PD, pBT)
            wl = torch.sum(E * J.unsqueeze(0), dim=(1, 2))
            factor = 0.5 if rA == rB else 1.0
            total_wl = total_wl + factor * wl
    return total_wl


def get_constraints_loss_with_regions(
    region_ps: Dict[str, torch.Tensor],
    alphas: Dict[str, float],
) -> torch.Tensor:
    """Constraint loss for N regions.  ``alphas`` maps region -> penalty weight."""
    total = 0
    for r, p in region_ps.items():
        usage = torch.sum(p, dim=1)
        coeff = p.shape[1] / 2.0 if r != 'io' else p.shape[1] / 20.0
        penalty = torch.sum(coeff * Func.softplus(usage - 1)**2, dim=1)
        total = total + alphas.get(r, 1.0) * penalty
    return total


def expected_fpga_placement_with_regions(
    region_ps, region_coupling, region_distances, alphas):
    """Combined HPWL + constraint loss for N regions."""
    hpwl = get_hpwl_loss_qubo_with_regions(region_ps, region_coupling, region_distances)
    constrain = get_constraints_loss_with_regions(region_ps, alphas)
    return hpwl + constrain


# =============================================================================
# Manual Gradient Functions
# =============================================================================

def manual_grad_hpwl_loss(p, W, D):
    """
    Compute manual gradient of HPWL loss.

    Args:
        p: Probability distribution [batch_size, num_instances, num_sites]
        W: Weight matrix [num_instances, num_instances]
        D: Distance matrix [num_sites, num_sites]

    Returns:
        h_grad: Gradient [batch_size, num_instances, num_sites]
    """
    batch_size, _, _ = p.shape

    PD = torch.matmul(p, D)
    W_batch = W.unsqueeze(0).expand(batch_size, -1, -1)
    h_grad = torch.bmm(W_batch, PD)

    return h_grad


def manual_grad_constraint_loss(p, lambda_constraint=30.0):
    """
    Compute manual gradient of constraint loss.

    Args:
        p: Probability distribution [batch_size, num_instances, num_sites]
        lambda_constraint: Constraint weight

    Returns:
        h_grad: Gradient [batch_size, num_instances, num_sites]
    """
    site_occupancy = torch.sum(p, dim=1)
    excess = site_occupancy - 1.0

    softplus_val = Func.softplus(excess)
    sigmoid_val = torch.sigmoid(excess)

    site_grad = 2 * lambda_constraint * softplus_val * sigmoid_val
    h_grad = site_grad.unsqueeze(1).repeat(1, p.shape[1], 1)

    return h_grad


def manual_grad_placement(p, J, site_coords_matrix, lambda_constraint=30.0):
    """
    Compute manual gradient of combined placement loss.

    Args:
        p: Probability distribution [batch_size, num_instances, num_sites]
        J: Coupling matrix [num_instances, num_instances]
        site_coords_matrix: Site coordinates [num_sites, 2]
        lambda_constraint: Constraint weight

    Returns:
        dE_dh: Gradient [batch_size, num_instances, num_sites]
    """
    batch_size, N, _ = p.shape

    # Distance matrix
    coords_i = site_coords_matrix.unsqueeze(1)
    coords_j = site_coords_matrix.unsqueeze(0)
    D = torch.sum(torch.abs(coords_i - coords_j), dim=2)

    # HPWL gradient
    PD = torch.matmul(p, D)
    mask = torch.triu(torch.ones(N, N, device=p.device), diagonal=1).bool()
    J_upper = J * mask
    J_batch = J_upper.unsqueeze(0).expand(batch_size, -1, -1)
    dE_hpwl_dp = torch.bmm(J_batch, PD)

    # Constraint gradient
    site_occupancy = torch.sum(p, dim=1)
    excess = site_occupancy - 1.0
    softplus_val = Func.softplus(excess)
    sigmoid_val = torch.sigmoid(excess)
    site_grad = 2 * lambda_constraint * softplus_val * sigmoid_val
    dE_constraint_dp = site_grad.unsqueeze(1).expand(-1, N, -1)

    dE_dp = dE_hpwl_dp + dE_constraint_dp

    # Compute softmax gradient
    sum_term = torch.sum(dE_dp * p, dim=2, keepdim=True)
    dE_dh = dE_dp * p - p * sum_term

    return dE_dh


# =============================================================================
# Expected Placement Loss Functions (QUBO approach)
# =============================================================================

def expected_fpga_placement(J, p, D, step, site_coords_matrix, alpha):
    """
    Calculate expected placement loss (HPWL + constraints).

    Args:
        J: Coupling matrix [num_instances, num_instances]
        p: Probability distribution [batch_size, num_instances, num_sites]
        site_coords_matrix: Site coordinates [num_sites, 2]
        step: Current optimization step (for history tracking)
        area_width: Width of the placement area
        alpha: Constraint weight

    Returns:
        total_loss: Combined loss [batch_size]
    """
    global _hpwl_loss_history, _constrain_loss_history, _total_loss_history, _placement_history

    hpwl = get_hpwl_loss_qubo(J, p, D)
    constrain_loss = get_constraints_loss(p, alpha)
    # constrain_loss = get_constraints_loss_rev(p, alpha)

    hpwl_val = hpwl
    total_val = hpwl_val + constrain_loss

    _hpwl_loss_history.append(hpwl_val.mean().item())
    _constrain_loss_history.append(constrain_loss.mean().item())
    _total_loss_history.append(total_val.mean().item())

    if step in show_steps:
        inst_indices = torch.argmax(p, dim=2)
        inst_coords = site_coords_matrix[inst_indices]
        _placement_history.append(inst_coords)

    return hpwl + constrain_loss


def expected_fpga_placement_with_io(J_LL, J_LI, 
                                    p_logic, p_io, 
                                    D_LL, D_LI,
                                    alpha, beta):
    """
    Calculate expected placement loss including IO connections.

    Args:
        J_LL: Logic-logic coupling matrix
        J_LI: Logic-IO coupling matrix
        p_logic: Logic probability distribution
        p_io: IO probability distribution
        D_LL: Logic-logic distance matrix
        D_LI: Logic-IO distance matrix

    Returns:
        total_loss: Combined loss [batch_size]
    """
    hpwl = get_hpwl_loss_qubo_with_io(J_LL, J_LI, p_logic, p_io, D_LL, D_LI)
    constrain_loss = get_constraints_loss_with_io(p_logic, p_io, alpha, beta)
    return hpwl + constrain_loss
    # return constrain_loss


# =============================================================================
# Inference Functions (QUBO approach)
# =============================================================================

def export_placement_qubo(F, site_coords_matrix, lam, mu, format='symmetric'):
    """
    Export the full QUBO matrix with slack variables for SB solver.

    Constructs a single Q matrix encoding:
        argmin_{x,s} ½ x^T (F⊗D) x + λ‖Ax - 1‖² + μ‖Bx - s‖²

    where x ∈ {0,1}^{mn} are placement variables,
          s ∈ {0,1}^n are slack variables (s_j=1 means site j is used).

    The at-most-one constraint uses equality form with slack:
        Σ_i x_{i,s} = s_s  for each site s
    This forces: site unused (s=0) or used by exactly one instance (s=1).

    Args:
        F: Coupling matrix [m, m] (flow/connectivity between instances)
        site_coords_matrix: Site coordinates [n, 2]
        lam: Weight for one-hot constraint (each instance picks exactly one site)
        mu: Weight for at-most-one constraint (each site used at most once)
        format: 'symmetric' (default) or 'upper_triangular'

    Returns:
        Q_full: QUBO matrix [(mn+n), (mn+n)]
        metadata: Dict with 'm', 'n', 'site_coords'
    """
    m = F.shape[0]
    n = site_coords_matrix.shape[0]
    device = F.device
    dtype = F.dtype

    D = get_site_distance_matrix(site_coords_matrix)

    # --- Q_xx block [mn × mn] ---
    # From HPWL: ½(F⊗D)
    # From one-hot  λ‖Ax-1‖²: λ(I_m⊗J_n) - 2λ·I_{mn}  (drops constant λ·m)
    # From at-most-one μ‖Bx-s‖²: μ(J_m⊗I_n)  (no linear term on x)
    ones_n = torch.ones(n, n, device=device, dtype=dtype)
    ones_m = torch.ones(m, m, device=device, dtype=dtype)
    I_n = torch.eye(n, device=device, dtype=dtype)
    I_m = torch.eye(m, device=device, dtype=dtype)
    I_mn = torch.eye(m * n, device=device, dtype=dtype)

    Q_xx = (0.5 * torch.kron(F, D)
            + lam * torch.kron(I_m, ones_n)
            + mu * torch.kron(ones_m, I_n)
            - 2 * lam * I_mn)

    # --- Q_xs block [mn × n] ---
    # From μ‖Bx-s‖²: cross term -2μ x^T B^T s → symmetric Q_xs = -μ·B^T
    # B = 1_m^T ⊗ I_n,  B^T = 1_m ⊗ I_n  ∈ R^{mn×n}
    B_T = torch.kron(torch.ones(m, 1, device=device, dtype=dtype), I_n)
    Q_xs = -mu * B_T

    # --- Q_ss block [n × n] ---
    # From μ‖Bx-s‖²: s^T s → (for binary s: s²=s) → +μ·I_n
    Q_ss = mu * I_n

    # Assemble full Q
    Q_top = torch.cat([Q_xx, Q_xs], dim=1)
    Q_bot = torch.cat([Q_xs.T, Q_ss], dim=1)
    Q_full = torch.cat([Q_top, Q_bot], dim=0)

    if format == 'upper_triangular':
        Q_full = torch.triu(Q_full) + torch.triu(Q_full, diagonal=1)

    metadata = {'m': m, 'n': n, 'site_coords': site_coords_matrix}
    return Q_full, metadata


def decode_qubo_solution(z, m, n, site_coords_matrix):
    """
    Decode a QUBO solution vector back into placement assignments.

    Args:
        z: Binary solution vector [mn + n] (x variables + slack variables)
        m: Number of instances
        n: Number of sites
        site_coords_matrix: Site coordinates [n, 2]

    Returns:
        site_indices: Assigned site index for each instance [m]
        coords: Coordinates for each instance [m, 2]
    """
    x = z[:m * n].reshape(m, n)
    site_indices = torch.argmax(x, dim=1)
    coords = site_coords_matrix[site_indices]
    return site_indices, coords


def solve_placement_sb(F, site_coords_matrix, lam=50.0, mu=50.0,
                       agents=128, max_steps=10000, best_only=True, **sb_kwargs):
    """
    Solve placement QUBO using the simulated-bifurcation library.

    Uses heated discrete SB mode (dSB) which works well for placement QUBOs.
    Heated mode adds annealing to help escape local optima where instances
    collapse to the same site.

    Args:
        F: Coupling matrix [m, m] (flow/connectivity between instances)
        site_coords_matrix: Site coordinates [n, 2]
        lam: Weight for one-hot constraint (each instance picks exactly one site)
        mu: Weight for at-most-one constraint (each site used at most once)
        agents: Number of SB agents (parallel runs)
        max_steps: Maximum SB iterations
        best_only: If True, return only the best solution
        **sb_kwargs: Additional keyword arguments passed to sb.minimize

    Returns:
        site_indices: Assigned site index for each instance [m]
        coords: Coordinates for each instance [m, 2]
        energy: Scalar energy of the best solution (QUBO energy)
        metadata: Dict with 'm', 'n', 'site_coords'
    """
    import simulated_bifurcation as sb

    Q, meta = export_placement_qubo(F, site_coords_matrix, lam, mu,
                                    format='symmetric')
    m, n = meta['m'], meta['n']

    # Use heated discrete mode - heated adds annealing to avoid collapsing
    # to degenerate solutions; ballistic mode fails due to large Ising fields
    sb_kwargs.setdefault('mode', 'discrete')
    sb_kwargs.setdefault('heated', True)
    if 'device' not in sb_kwargs and torch.cuda.is_available():
        sb_kwargs['device'] = 'cuda'

    z, energy = sb.minimize(Q, domain='binary', agents=agents,
                            max_steps=max_steps, best_only=best_only,
                            **sb_kwargs)
    z_tensor = z if isinstance(z, torch.Tensor) else torch.tensor(z, dtype=Q.dtype)
    z_tensor = z_tensor.cpu()

    site_indices, coords = decode_qubo_solution(z_tensor.float(), m, n,
                                                meta['site_coords'])
    return site_indices, coords, energy, meta


# =============================================================================
# CyclicExpansion Placement Solver (arXiv:2312.15467)
# =============================================================================

def _qap_cost(F, D, perm):
    """Compute QAP cost: sum_{i<j} F[i,j] * D[perm[i], perm[j]].

    Args:
        F: Flow matrix [m, m] (symmetric)
        D: Distance matrix [n, n]
        perm: Permutation vector [m] mapping instances to sites

    Returns:
        Scalar cost value
    """
    D_sub = D[perm][:, perm]  # [m, m]
    # Only upper triangle to avoid double-counting
    m = F.shape[0]
    mask = torch.triu(torch.ones(m, m, device=F.device, dtype=torch.bool), diagonal=1)
    return (F[mask] * D_sub[mask]).sum()


def _single_swap_delta(F, D, perm, i1, i2):
    """O(m) cost change from swapping sites of instances i1 and i2.

    Args:
        F: Flow matrix [m, m]
        D: Distance matrix [n, n]
        perm: Current permutation [m]
        i1, i2: Instance indices to swap

    Returns:
        Scalar delta (new_cost - old_cost)
    """
    a, b = perm[i1], perm[i2]
    f_diff = F[i1, :] - F[i2, :]           # [m]
    d_diff = D[b, perm] - D[a, perm]       # [m]
    # Exclude self-interactions at i1 and i2
    delta = (f_diff * d_diff).sum() - f_diff[i1] * d_diff[i1] - f_diff[i2] * d_diff[i2]
    return delta


def _single_move_delta(F, D, perm, i, j_new):
    """O(m) cost change from moving instance i to unbound site j_new.

    Args:
        F: Flow matrix [m, m]
        D: Distance matrix [n, n]
        perm: Current permutation [m]
        i: Instance index to move
        j_new: New site index

    Returns:
        Scalar delta (new_cost - old_cost)
    """
    a = perm[i]
    d_diff = D[j_new, perm] - D[a, perm]   # [m]
    delta = (F[i, :] * d_diff).sum() - F[i, i] * d_diff[i]
    return delta


def _cycle_interaction(F, D, perm, cycle1, cycle2):
    """O(1) interaction energy between two disjoint 2-cycles.

    For two swap cycles (i1<->i2) and (i3<->i4), computes the
    additional cost change when both are applied simultaneously
    vs the sum of individual deltas.

    Args:
        F: Flow matrix [m, m]
        D: Distance matrix [n, n]
        perm: Current permutation [m]
        cycle1, cycle2: Tuples describing cycles

    Returns:
        Scalar interaction term
    """
    # Extract the instance indices involved in each cycle
    if cycle1[0] == 'swap':
        i1, i2 = cycle1[1], cycle1[2]
        a1_old, a1_new = perm[i1].item(), perm[i2].item()
        b1_old, b1_new = perm[i2].item(), perm[i1].item()
        insts1 = [(i1, a1_old, a1_new), (i2, b1_old, b1_new)]
    else:  # move
        i1, j_new = cycle1[1], cycle1[2]
        a1_old = perm[i1].item()
        insts1 = [(i1, a1_old, j_new)]

    if cycle2[0] == 'swap':
        i3, i4 = cycle2[1], cycle2[2]
        a2_old, a2_new = perm[i3].item(), perm[i4].item()
        b2_old, b2_new = perm[i4].item(), perm[i3].item()
        insts2 = [(i3, a2_old, a2_new), (i4, b2_old, b2_new)]
    else:  # move
        i3, j_new2 = cycle2[1], cycle2[2]
        a2_old = perm[i3].item()
        insts2 = [(i3, a2_old, j_new2)]

    # Interaction = sum over (inst_a in cycle1, inst_b in cycle2) of
    # F[ia,ib] * (D[new_a, new_b] - D[new_a, old_b] - D[old_a, new_b] + D[old_a, old_b])
    interaction = 0.0
    for (ia, old_a, new_a) in insts1:
        for (ib, old_b, new_b) in insts2:
            interaction += F[ia, ib] * (
                D[new_a, new_b] - D[new_a, old_b]
                - D[old_a, new_b] + D[old_a, old_b]
            )
    return interaction


def _sample_disjoint_cycles(perm, n, k, k_u, rng):
    """Sample disjoint 2-cycles from instances and unbound sites.

    Args:
        perm: Current permutation [m]
        n: Total number of sites
        k: Number of instances to consider for swaps
        k_u: Number of unbound sites to consider for moves
        rng: numpy random Generator

    Returns:
        List of cycle tuples: ('swap', i1, i2) or ('move', i, j_site)
    """
    m = len(perm)

    # Select subset of instances
    inst_subset = rng.choice(m, size=min(k, m), replace=False)
    rng.shuffle(inst_subset)

    # Collect unbound sites
    bound_sites = set(perm.tolist())
    unbound_sites = [s for s in range(n) if s not in bound_sites]

    # Sample unbound sites
    if len(unbound_sites) > k_u:
        ub_indices = rng.choice(len(unbound_sites), size=k_u, replace=False)
        unbound_sample = [unbound_sites[i] for i in ub_indices]
    else:
        unbound_sample = list(unbound_sites)

    cycles = []
    used_insts = set()
    used_sites = set()

    # Generate swap cycles from pairs of instances
    for idx in range(0, len(inst_subset) - 1, 2):
        i1, i2 = int(inst_subset[idx]), int(inst_subset[idx + 1])
        if i1 in used_insts or i2 in used_insts:
            continue
        cycles.append(('swap', i1, i2))
        used_insts.add(i1)
        used_insts.add(i2)

    # Generate move cycles from remaining instances to unbound sites
    remaining_insts = [int(i) for i in inst_subset if i not in used_insts]
    for i, j in zip(remaining_insts, unbound_sample):
        if i in used_insts or j in used_sites:
            continue
        cycles.append(('move', i, j))
        used_insts.add(i)
        used_sites.add(j)

    return cycles


def _build_cyclic_sub_qubo(F, D, perm, cycles):
    """Build small s x s QUBO from disjoint cycles.

    Q[i,i] = delta_i (single cycle cost change)
    Q[i,j] = interaction(cycle_i, cycle_j) for i != j

    Args:
        F: Flow matrix [m, m]
        D: Distance matrix [n, n]
        perm: Current permutation [m]
        cycles: List of cycle tuples

    Returns:
        Q: QUBO matrix [s, s] as torch tensor
    """
    s = len(cycles)
    Q = torch.zeros(s, s, dtype=F.dtype, device=F.device)

    # Diagonal: single cycle deltas
    for i, cyc in enumerate(cycles):
        if cyc[0] == 'swap':
            Q[i, i] = _single_swap_delta(F, D, perm, cyc[1], cyc[2])
        else:  # move
            Q[i, i] = _single_move_delta(F, D, perm, cyc[1], cyc[2])

    # Off-diagonal: pairwise interactions
    for i in range(s):
        for j in range(i + 1, s):
            val = _cycle_interaction(F, D, perm, cycles[i], cycles[j])
            Q[i, j] = val
            Q[j, i] = val

    return Q


def _apply_cycles(perm, cycles, alpha):
    """Apply selected cycles to permutation.

    Args:
        perm: Current permutation [m] (modified in-place)
        cycles: List of cycle tuples
        alpha: Binary selection vector [s]

    Returns:
        perm (modified in-place)
    """
    for i, cyc in enumerate(cycles):
        if alpha[i] == 0:
            continue
        if cyc[0] == 'swap':
            i1, i2 = cyc[1], cyc[2]
            perm[i1], perm[i2] = perm[i2].clone(), perm[i1].clone()
        else:  # move
            inst, site = cyc[1], cyc[2]
            perm[inst] = site
    return perm


def _solve_sub_qubo_greedy(Q):
    """Solve small binary QUBO by greedy bit-flip.

    For sub-QUBOs of size s <= 30, try each variable greedily.

    Args:
        Q: QUBO matrix [s, s]

    Returns:
        alpha: Binary solution vector [s]
    """
    s = Q.shape[0]
    alpha = torch.zeros(s, dtype=Q.dtype, device=Q.device)

    # Greedy: consider flipping each bit
    for _ in range(2):  # Two passes for better quality
        for i in range(s):
            # Cost of flipping bit i
            # delta = Q[i,i] + 2 * sum_j(Q[i,j] * alpha[j]) for j != i
            delta = Q[i, i] + 2.0 * (Q[i, :] * alpha).sum() - 2.0 * Q[i, i] * alpha[i]
            if alpha[i] == 0 and delta < 0:
                alpha[i] = 1.0
            elif alpha[i] == 1 and -delta + Q[i, i] > 0:
                # Unflip if removing this bit improves cost
                # Cost of turning off: -(Q[i,i] + 2*sum_{j!=i} Q[i,j]*alpha[j])
                cost_off = -(Q[i, i] + 2.0 * ((Q[i, :] * alpha).sum() - Q[i, i] * alpha[i]))
                if cost_off < 0:
                    alpha[i] = 0.0

    return alpha


def solve_placement_cyclic(
    F, site_coords_matrix,
    k=60, k_u=30, max_iters=50,
    init_perm=None, seed=None, verbose=False,
):
    """Solve placement using the CyclicExpansion algorithm.

    Iteratively samples small 2-cycle swap sub-problems and solves them
    to incrementally improve placement. 2-cycles naturally preserve
    permutation feasibility without one-hot/at-most-one constraints.

    Based on arXiv:2312.15467.

    Args:
        F: Flow/coupling matrix [m, m] (symmetric)
        site_coords_matrix: Site coordinates [n, 2]
        k: Number of instances to consider per iteration for swaps
        k_u: Number of unbound sites to consider per iteration for moves
        max_iters: Maximum number of iterations
        init_perm: Initial permutation [m] (optional, random if None)
        seed: Random seed for reproducibility
        verbose: Print progress information

    Returns:
        site_indices: Best site index for each instance [m]
        coords: Coordinates for each instance [m, 2]
        energy: Best QAP cost (scalar)
        metadata: Dict with 'm', 'n', 'cost_history', 'iterations'
    """
    import numpy as np

    m = F.shape[0]
    n = site_coords_matrix.shape[0]
    assert m <= n, f"Need m <= n, got m={m}, n={n}"

    device = F.device

    # Compute distance matrix
    D = get_site_distance_matrix(site_coords_matrix)

    # Make F symmetric with zero diagonal
    F = (F + F.T) / 2
    F = F.clone()
    F.fill_diagonal_(0)

    rng = np.random.default_rng(seed)

    # Initialize permutation
    if init_perm is not None:
        perm = init_perm.clone().long()
    else:
        perm_np = rng.choice(n, size=m, replace=False)
        perm = torch.tensor(perm_np, device=device, dtype=torch.long)

    # Compute initial cost
    current_cost = _qap_cost(F, D, perm).item()
    best_cost = current_cost
    best_perm = perm.clone()
    cost_history = [best_cost]

    patience_counter = 0
    patience = 5

    for it in range(max_iters):
        # Sample disjoint cycles
        cycles = _sample_disjoint_cycles(perm, n, k, k_u, rng)

        if len(cycles) == 0:
            if verbose:
                WARNING(f"Iteration {it}: no cycles sampled, skipping")
            continue

        # Build sub-QUBO
        Q_sub = _build_cyclic_sub_qubo(F, D, perm, cycles)

        # Solve sub-QUBO
        alpha = _solve_sub_qubo_greedy(Q_sub)

        # Apply selected cycles
        if alpha.sum() > 0:
            perm = _apply_cycles(perm, cycles, alpha)

        # Recompute cost (vectorized O(m²) torch op, not the bottleneck)
        current_cost = _qap_cost(F, D, perm).item()
        cost_history.append(current_cost)

        if current_cost < best_cost - 1e-10:
            best_cost = current_cost
            best_perm = perm.clone()
            patience_counter = 0
        else:
            patience_counter += 1

        if verbose:
            n_applied = int(alpha.sum().item())
            INFO(f"Iter {it}: cost={current_cost:.4f}, best={best_cost:.4f}, "
                  f"cycles={len(cycles)}, applied={n_applied}")

        if patience_counter >= patience:
            if verbose:
                INFO(f"Early stop at iter {it} (patience={patience})")
            break

    # Return best solution
    site_indices = best_perm
    coords = site_coords_matrix[site_indices]
    metadata = {
        'm': m, 'n': n,
        'site_coords': site_coords_matrix,
        'cost_history': cost_history,
        'iterations': len(cost_history) - 1,
    }

    return site_indices, coords, best_cost, metadata


def infer_placements(J, p, logic_site_coords, D):
    """
    Infer final placements from probability distribution (matches master exactly).

    Args:
        J: Coupling matrix [num_instances, num_instances]
        p: Probability distribution [batch_size, num_instances, num_sites]
        logic_site_coords: Site coordinates [num_sites, 2]
        D: Distance matrix [num_sites, num_sites]

    Returns:
        inst_coords: Instance coordinates [batch_size, num_instances, 2]
        hpwl: HPWL values [batch_size]
    """
    inst_indices = torch.argmax(p, dim=2)
    inst_coords = logic_site_coords[inst_indices]
    result = get_hpwl_loss_qubo(J, p, D)
    return inst_coords, result


def infer_placements_with_io(J_LL, J_LI, 
                             p_logic, p_io, 
                             logic_site_coords, 
                             D_LL, D_LI, io_site_coords):
    """
    Infer final placements including IO from probability distributions.

    Args:
        J_LL: Logic-logic coupling matrix
        J_LI: Logic-IO coupling matrix
        p_logic: Logic probability distribution
        p_io: IO probability distribution
        logic_site_coords: Logic site coordinates [num_logic_sites, 2]
        D_LL, D_LI: Distance matrices
        io_site_coords: Valid site coordinates for IO.

    Returns:
        coords: List of [logic_coords, io_coords]
        hpwl: HPWL values [batch_size]
    """
    logic_inst_indices = torch.argmax(p_logic, dim=2)
    io_inst_indices = torch.argmax(p_io, dim=2)
    
    logic_inst_coords = logic_site_coords[logic_inst_indices]
    io_inst_coords = io_site_coords[io_inst_indices]

    logic_overlap, _, _ = count_duplicate_coords(logic_inst_coords)
    io_overlap, _, _ = count_duplicate_coords(io_inst_coords)
    INFO(f"Logic overlap: {logic_overlap}, IO overlap: {io_overlap}")

    result = get_hpwl_loss_qubo_with_io(J_LL, J_LI, 
                                        p_logic, p_io, 
                                        D_LL, D_LI)
    return [logic_inst_coords, io_inst_coords], result


def infer_placements_with_regions(
    region_ps: Dict[str, torch.Tensor],
    region_site_coords: Dict[str, torch.Tensor],
    region_coupling: Dict[str, Dict[str, torch.Tensor]],
    region_distances: Dict[str, Dict[str, torch.Tensor]],
) -> tuple:
    """
    Infer final placements for N regions from probability distributions.

    Args:
        region_ps: Region name → probability tensor [B, N_i, M_i]
        region_site_coords: Region name → site coords [M_i, 2]
        region_coupling: (rA, rB) → coupling matrix [N_A, N_B]
        region_distances: (rA, rB) → distance matrix [M_A, M_B]

    Returns:
        coords: Dict[r] → instance coords [B, N_i, 2]
        hpwl: HPWL values [B]
    """
    coords = {}
    for r, p in region_ps.items():
        inst_indices = torch.argmax(p, dim=2)
        coords[r] = region_site_coords[r][inst_indices]

        overlap, _, _ = count_duplicate_coords(coords[r])
        if overlap > 0:
            INFO(f"  {r} overlap: {overlap}")

    result = get_hpwl_loss_qubo_with_regions(region_ps, region_coupling, region_distances)
    return coords, result
