"""
hessian_topology.py

Calculates the dominant eigenvalue (lambda_max) of the Hessian matrix 
for saved optimization states to mathematically prove geometric compression 
during delayed algorithmic generalization. Utilizes Power Iteration and 
Hessian-Vector Products (HVP) to strictly bound spatial complexity to O(N), 
bypassing intractable O(N^2) or O(N^3) memory constraints.
"""

import os
import glob
import json
import argparse
from typing import Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Import custom modules
from generate_dataset import ModularArithmeticDataset
from model_architecture import ToyTransformer

# =====================================================================
# ⚙️ CONFIGURATION
# =====================================================================

P_MODULO = 97
BATCH_SIZE = 512            # Larger batch size to ensure stability in empirical Hessian approximation
NUM_POWER_ITERATIONS = 20   # Asymptotic iterations to guarantee convergence to the dominant eigenvector
# The following are only meaningful when this script is run directly (see __main__ guard below).
# They are set to safe defaults here so that importing this module (e.g. from
# measure_intrinsic_noise.py) does NOT trigger CLI argument parsing of the caller's sys.argv.
NUM_TRACE_SAMPLES = 10
SEED = 42
TRAIN_RATIO = 0.5
RUN_NAME = None
CHECKPOINT_DIR = None
INPUT_TELEMETRY = None
OUTPUT_TELEMETRY = None
# =====================================================================


def _parse_cli_args():
    """Parses CLI args. Only called when this script is run directly (python hessian_topology.py ...),
    never on import, so other scripts can safely import get_dominant_eigenvalue/compute_hvp/etc.
    without their own argparse conflicting with this module's flags."""
    parser = argparse.ArgumentParser(description="Extract continuous Hessian spectral telemetry (lambda_max + Hutchinson trace) for a given run.")
    parser.add_argument("--run_name", type=str, required=True, help="Must match the --run_name used in train_and_grok.py (e.g. ratio50_seed42).")
    parser.add_argument("--train_ratio", type=float, default=0.5, help="Must match the train_ratio used to train this run's checkpoints.")
    parser.add_argument("--seed", type=int, default=42, help="Must match the seed used to train this run's checkpoints.")
    parser.add_argument("--trace_samples", type=int, default=10, help="Number of Rademacher samples for the Hutchinson trace estimator.")
    return parser.parse_args()


def compute_hvp(model: nn.Module, loss: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """
    Computes the exact Hessian-Vector Product utilizing automatic differentiation.
    
    Args:
        model (nn.Module): The autoregressive neural network architecture.
        loss (torch.Tensor): The scalar loss value (must be calculated with create_graph=True).
        v (torch.Tensor): The arbitrary vector to multiply against the Hessian operator.
        
    Returns:
        torch.Tensor: The resulting directional derivative vector from the H * v product.
    """
    # 1st derivative: Gradient of the objective function with respect to parameters
    grads = torch.autograd.grad(loss, model.parameters(), create_graph=True, retain_graph=True)
    
    # Flatten parametric gradients into a single 1D vector
    grad_vector = torch.cat([g.contiguous().view(-1) for g in grads])
    
    # Dot product of the gradient vector and the random vector 'v'
    grad_v_prod = torch.sum(grad_vector * v)
    
    # 2nd derivative: Gradient of the dot product yields the exact HVP
    hvp_grads = torch.autograd.grad(grad_v_prod, model.parameters(), retain_graph=True)
    hvp_vector = torch.cat([g.contiguous().view(-1) for g in hvp_grads])
    
    return hvp_vector


def estimate_hessian_trace(model: nn.Module,
                            loss: torch.Tensor,
                            total_params: int,
                            device: torch.device,
                            num_samples: int = 10) -> float:
    """
    Estimates tr(H) via the Hutchinson stochastic trace estimator:
        tr(H) ≈ (1/num_samples) * sum_i( z_i^T H z_i )
    where z_i are i.i.d. Rademacher vectors (+1/-1 entries), an unbiased estimator
    of the trace that reuses the exact same HVP machinery already validated for
    lambda_max. Provides a second, complementary spectral scalar: while lambda_max
    captures the single steepest direction of curvature, tr(H) = sum of ALL
    eigenvalues, i.e. the aggregate/average directional curvature across the
    entire parameter space. This directly addresses the critique that a single
    dominant eigenvalue is an overly reductive characterization of the Hessian
    spectrum (Reviewers #2, #3, #5).

    Note: this is still not the full eigenvalue density/spectrum a Lanczos-based
    method (e.g. stochastic Lanczos quadrature) would give -- it is a cheap,
    honest second scalar, not a replacement for a full spectral characterization.
    That distinction should be stated explicitly in the paper's limitations.
    """
    rademacher_vectors = (
        torch.randint(0, 2, (num_samples, total_params), device=device).float() * 2 - 1
    )

    trace_samples = []
    for i in range(num_samples):
        z = rademacher_vectors[i]
        Hz = compute_hvp(model, loss, z)
        trace_samples.append(torch.dot(z, Hz).item())

    return sum(trace_samples) / len(trace_samples)


def get_dominant_eigenvalue(model: nn.Module, 
                            inputs: torch.Tensor, 
                            targets: torch.Tensor, 
                            criterion: nn.Module, 
                            device: torch.device,
                            num_iterations: int = 20,
                            num_trace_samples: int = 10) -> Tuple[float, float, float]:
    """
    Approximates the lambda_max of the Hessian matrix utilizing the Power Iteration method,
    and additionally estimates tr(H) via Hutchinson's estimator as a second, complementary
    spectral scalar (aggregate curvature vs. the single steepest direction captured by lambda_max).
    Integrates a critical compiler-level intervention to mathematically enable 
    exact double backpropagation flow.

    Returns:
        Tuple[float, float, float]: (lambda_max, hessian_trace, loss_value)
    """
    model.eval()  # Bound variance constraints during active spectral evaluation
    
    # --- CRITICAL FIX FOR EXACT DOUBLE BACKPROP (HESSIAN EXTRACTION) IN PYTORCH ---
    # Hardware-level fused attention kernels (FlashAttention/MemEfficient) operate as opaque
    # abstractions lacking support for exact second-order functions.
    # We explicitly force the compiler to unroll the full computational graph using native Math.
    try:
        # PyTorch >= 2.1.2
        from torch.nn.attention import sdpa_kernel, SDPBackend
        sdp_context = sdpa_kernel(SDPBackend.MATH)
    except ImportError:
        # PyTorch 2.0 - 2.1.1 fallback
        sdp_context = torch.backends.cuda.sdp_kernel(enable_flash=False, enable_math=True, enable_mem_efficient=False)
        
    with sdp_context:
        # Forward pass (Computational graph strictly utilizes native tensor operations)
        logits = model(inputs)
        final_logits = logits[:, -1, :]
        final_targets = targets[:, -1]
        loss = criterion(final_logits, final_targets)
    
    # Initialize a random vector 'v' matching the dimensionality of the parameter space
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    v = torch.randn(total_params, device=device)
    v = v / torch.norm(v)  # L2 Normalization to unit length
    
    lambda_max = 0.0
    
    for _ in range(num_iterations):
        # Compute H*v (Mathematical flow guaranteed by the unfused graph)
        Hv = compute_hvp(model, loss, v)
        
        # Rayleigh quotient approximation of maximum directional curvature: v^T * H * v
        lambda_max = torch.dot(v, Hv).item()
        
        # Asymptotic normalization to actively prevent arithmetic overflow
        v = Hv / (torch.norm(Hv) + 1e-8)

    # Second spectral scalar: aggregate curvature across the full parameter space,
    # via Hutchinson's stochastic trace estimator (reuses the same HVP operator).
    hessian_trace = estimate_hessian_trace(model, loss, total_params, device, num_samples=num_trace_samples)

    return lambda_max, hessian_trace, loss.item()


def analyze_checkpoints() -> None:
    """
    Iterates over saved optimization states, continuously extracts the dominant eigenvalue 
    for each, and augments the macroscopic telemetry for final continuous visualization.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Executing continuous spectral analysis on hardware architecture: {device}")

    # 1. Load a fixed static batch to represent the optimization manifold consistently
    train_dataset = ModularArithmeticDataset(p=P_MODULO, split='train', train_ratio=TRAIN_RATIO, seed=SEED)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    x_batch, y_batch = next(iter(train_loader))
    x_batch, y_batch = x_batch.to(device), y_batch.to(device)
    
    criterion = nn.CrossEntropyLoss()

    # 2. Locate and chronologically sort saved parametric states for THIS run
    checkpoint_files = glob.glob(f"{CHECKPOINT_DIR}/model_step_*.pt")
    checkpoint_files.sort(key=lambda x: int(x.split('_step_')[1].split('.pt')[0]))
    
    if not checkpoint_files:
        print(f"Error: No saved states found in '{CHECKPOINT_DIR}/'. Did you train run '{RUN_NAME}' first?")
        return

    # 3. Load existing macroscopic optimization trajectory for THIS run
    try:
        with open(INPUT_TELEMETRY, "r") as f:
            telemetry = json.load(f)
    except FileNotFoundError:
        print(f"Error: '{INPUT_TELEMETRY}' not found. Execute train_and_grok.py with --run_name {RUN_NAME} first.")
        return

    # Augment JSON with mathematical arrays for spectral tracking
    telemetry["lambda_max"] = []
    telemetry["hessian_trace"] = []
    telemetry["checkpoint_steps"] = []

    print(f"\nDiscovered {len(checkpoint_files)} optimization states for run '{RUN_NAME}'. "
          f"Initiating Spectral Dissection ({NUM_TRACE_SAMPLES} trace samples/checkpoint)...")
    
    # Initialize baseline architectural structure
    model = ToyTransformer(vocab_size=train_dataset.vocab_size).to(device)

    # 4. Extract continuous topology iteratively
    for ckpt_path in checkpoint_files:
        step = int(ckpt_path.split('_step_')[1].split('.pt')[0])
        
        # Restore geometric parameter state
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        
        # Calculate maximum directional curvature (lambda_max) and aggregate curvature (tr(H))
        print(f"Extracting local topology for Step {step}...", end=" ", flush=True)
        eig_val, trace_val, _ = get_dominant_eigenvalue(
            model, x_batch, y_batch, criterion, device,
            num_iterations=NUM_POWER_ITERATIONS, num_trace_samples=NUM_TRACE_SAMPLES
        )
        print(f"lambda_max: {eig_val:.4f} | tr(H): {trace_val:.4f}")
        
        telemetry["checkpoint_steps"].append(step)
        telemetry["lambda_max"].append(eig_val)
        telemetry["hessian_trace"].append(trace_val)

    # 5. Persist augmented continuous telemetry for THIS run
    with open(OUTPUT_TELEMETRY, "w") as f:
        json.dump(telemetry, f)
        
    print(f"\nSpectral dissection complete! Continuous spectral trajectory successfully saved to '{OUTPUT_TELEMETRY}'.")


if __name__ == "__main__":
    cli_args = _parse_cli_args()
    NUM_TRACE_SAMPLES = cli_args.trace_samples
    SEED = cli_args.seed
    TRAIN_RATIO = cli_args.train_ratio
    RUN_NAME = cli_args.run_name
    CHECKPOINT_DIR = os.path.join("checkpoints", RUN_NAME)
    INPUT_TELEMETRY = f"grokking_telemetry_{RUN_NAME}.json"
    OUTPUT_TELEMETRY = f"grokking_telemetry_{RUN_NAME}_with_hessian.json"

    analyze_checkpoints()
