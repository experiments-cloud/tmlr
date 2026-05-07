"""
generate_dataset.py

Generates the strictly synthetic algebraic dataset for delayed algorithmic 
generalization experiments. Task: Discrete Modular Addition a + b (mod p).

This module isolates the topological phase transition by forcing the architecture
to consolidate a cyclical mathematical algorithm (modulo arithmetic) rather than 
overfitting to heuristic pairs. It is designed to formally evaluate both standard 
generalization and the structural limits of learning via data scarcity ablation.
"""

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

class ModularArithmeticDataset(Dataset):
    """
    Algorithmic dataset generator for the discrete modular arithmetic task: a + b (mod p).
    """
    
    def __init__(self, p: int = 97, split: str = 'train', train_ratio: float = 0.5, seed: int = 42):
        """
        Initializes the dataset and performs deterministic orthogonal structural partitions.

        =====================================================================
        🔬 TOPOLOGICAL ABLATION STUDY CONFIGURATION (Ref: Section 5.3)
        =====================================================================
        Modify the 'train_ratio' parameter to analytically reproduce the structural 
        boundary limits of algorithmic consolidation:
        
        * train_ratio = 0.50 (Base) -> Relative abundance. Rapid algorithmic consolidation.
        * train_ratio = 0.25 (Grok) -> Moderate data scarcity. Induces standard delayed 
                                       algorithmic generalization.
        * train_ratio = 0.10 (Fail) -> Severe data scarcity. Architecture remains trapped 
                                       in a high-curvature, uncompressed minimum.
        =====================================================================

        Args:
            p (int): Prime number defining the finite cyclic group (Z_p). Default is 97.
            split (str): 'train' for training subset, 'val' for validation subset.
            train_ratio (float): Fraction of the total data manifold used for training. 
            seed (int): Random seed for strict reproducibility across topological regimes.
        """
        self.p = p
        self.split = split
        self.vocab_size = p + 3
        
        # Discrete algorithmic tokens
        self.OP_TOKEN = p       # Represents the '+' operator
        self.EQ_TOKEN = p + 1   # Represents the '=' operator
        self.EOS_TOKEN = p + 2  # Represents End-Of-Sequence
        
        # Generate the complete optimization manifold of (a, b) combinations (Total: p^2)
        a_vals = torch.arange(p)
        b_vals = torch.arange(p)
        grid_a, grid_b = torch.meshgrid(a_vals, b_vals, indexing='ij')
        
        self.a_data = grid_a.flatten()
        self.b_data = grid_b.flatten()
        
        # Evaluate the exact mathematical mapping: c = (a + b) % p
        self.c_data = (self.a_data + self.b_data) % self.p
        
        # Deterministic structural partitioning (Train/Validation)
        total_samples = len(self.a_data)
        indices = np.arange(total_samples)
        
        # Enforces strict mathematical orthogonality: the architecture must generalize 
        # to structurally unseen boundary conditions
        np.random.seed(seed)
        np.random.shuffle(indices)
        
        train_size = int(total_samples * train_ratio)
        
        if split == 'train':
            self.indices = indices[:train_size]
        else:
            self.indices = indices[train_size:]
            
        print(f"Dataset '{split}' successfully initialized with {len(self.indices)} samples "
              f"(Density Ratio: {train_ratio * 100}%). Modulo p={self.p}.")

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int):
        real_idx = self.indices[idx]
        a = self.a_data[real_idx]
        b = self.b_data[real_idx]
        c = self.c_data[real_idx]
        
        # Strict autoregressive tensor formatting (Teacher Forcing)
        # Input tensor X algebraically structured as: [a, +, b, =]
        # Shifted target tensor Y: [+, b, =, c] (Right-shifted for causal next-token prediction)
        x = torch.tensor([a, self.OP_TOKEN, b, self.EQ_TOKEN], dtype=torch.long)
        y = torch.tensor([self.OP_TOKEN, b, self.EQ_TOKEN, c], dtype=torch.long)
        
        return x, y

# --- Execution and Independent Analytical Verification ---
if __name__ == "__main__":
    # Example execution demonstrating the 25% topological ablation setup
    print("--- Executing Analytical Verification ---")
    train_dataset = ModularArithmeticDataset(p=97, split='train', train_ratio=0.25)
    val_dataset = ModularArithmeticDataset(p=97, split='val', train_ratio=0.25)
    
    # A static batch size of 256 provides optimal spatial scaling for the optimization trajectory
    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
    
    x_batch, y_batch = next(iter(train_loader))
    
    print("\n--- Topological Dataset Telemetry ---")
    print(f"Input tensor X shape: {x_batch.shape} -> (Batch_size, Sequence_length)")
    print(f"Target tensor Y shape: {y_batch.shape}")
    print(f"\nDecoded algebraic sequence from the initial batch:")
    print(f"Input X: [a={x_batch[0][0]}, op={x_batch[0][1]}, b={x_batch[0][2]}, eq={x_batch[0][3]}]")
    print(f"Target Y (to predict): {y_batch[0]}")
