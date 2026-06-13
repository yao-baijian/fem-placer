import time
import torch
import sys
import os

# Assuming fem_placer is in python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from fem_placer.objectives import get_hpwl_loss_qubo, get_hpwl_loss_qubo_sparse_accel

def test_hpwl_speed():
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    batch_size = 10
    num_instances = 1000
    num_sites = 200
    
    print(f"Batch Size: {batch_size}, Instances: {num_instances}, Sites: {num_sites}")
    
    # 1. Create dense and sparse J
    # J is symmetric coupling matrix
    J_dense = torch.randn(num_instances, num_instances, device=device)
    J_dense = (J_dense + J_dense.t()) * 0.5  # Make symmetric
    
    # Create sparse J
    # 5% sparsity
    mask = torch.rand(num_instances, num_instances, device=device) < 0.05
    J_sparse = (J_dense * mask).to_sparse()
    J_dense_sparse = J_dense * mask # Dense representation of sparse matrix for fair comparison of values
    
    # 2. Probability distribution p [batch, inst, sites]
    p = torch.randn(batch_size, num_instances, num_sites, device=device)
    p = torch.softmax(p, dim=2)
    
    # 3. Distance matrix D [sites, sites]
    D = torch.randn(num_sites, num_sites, device=device)
    D = (D + D.t()) * 0.5
    
    # --- Test 1: Original Dense ---
    start_time = time.time()
    for _ in range(10):
        loss_dense = get_hpwl_loss_qubo(J_dense_sparse, p, D)
    torch.cuda.synchronize() if device == 'cuda' else None
    end_time = time.time()
    dense_time = (end_time - start_time) / 10
    print(f"Original Dense Avg Time: {dense_time * 1000:.4f} ms")
    
    # --- Test 2: Sparse Accel ---
    start_time = time.time()
    for _ in range(10):
        loss_sparse = get_hpwl_loss_qubo_sparse_accel(J_sparse, p, D)
    torch.cuda.synchronize() if device == 'cuda' else None
    end_time = time.time()
    sparse_time = (end_time - start_time) / 10
    print(f"Sparse Accel Avg Time:   {sparse_time * 1000:.4f} ms")
    
    # --- Check Results ---
    diff = torch.abs(loss_dense - loss_sparse).max().item()
    print(f"Max Difference: {diff:.6e}")
    if diff < 1e-4:
        print("Verification: PASSED")
    else:
        print("Verification: FAILED")
        
    print("-" * 30)

if __name__ == "__main__":
    test_hpwl_speed()
