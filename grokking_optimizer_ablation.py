"""
grokking_optimizer_ablation.py

Executes a topological ablation study comparing the decoupled AdamW operator 
against standard coupled optimizers (Adam, SGD). Evaluates the continuous 
macroscopic and spectral dynamics to analytically demonstrate that the geometric 
pressure from decoupled weight decay is a strictly necessary boundary condition 
to traverse the topological barrier and induce algorithmic consolidation.
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import itertools

# ==========================================
# 1. ALGORITHMIC DATASET GENERATION
# ==========================================
class ModularAdditionDataset(Dataset):
    def __init__(self, p=97, split='train', train_ratio=0.5, seed=42):
        """
        Discrete modular arithmetic dataset a + b = c (mod p).
        Vocabulary (0 to p-1) for integers, p for '+', p+1 for '='.
        """
        self.p = p
        torch.manual_seed(seed) # Maintain static seed for rigorous topological control
        
        # Generate the complete optimization manifold of equations (p^2)
        all_pairs = torch.cartesian_prod(torch.arange(p), torch.arange(p))
        indices = torch.randperm(len(all_pairs))
        all_pairs = all_pairs[indices]
        
        # Orthogonal 50/50 structural split
        split_idx = int(len(all_pairs) * train_ratio)
        self.data = all_pairs[:split_idx] if split == 'train' else all_pairs[split_idx:]
        
        self.plus_token = p
        self.eq_token = p + 1

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        a, b = self.data[idx]
        c = (a + b) % self.p
        # Input Tensor X: [a, +, b, =]
        x = torch.tensor([a, self.plus_token, b, self.eq_token], dtype=torch.long)
        # Shifted Target Tensor Y: [+, b, =, c]
        y = torch.tensor([self.plus_token, b, self.eq_token, c], dtype=torch.long)
        return x, y

# ==========================================
# 2. CAUSAL TRANSFORMER ARCHITECTURE
# ==========================================
class SmallCausalTransformer(nn.Module):
    def __init__(self, vocab_size=100, d_model=128, n_heads=4, n_layers=2, d_ff=512):
        """
        Transformer architecture configured with N ~ 422,000 parameters and Pre-LN.
        """
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = nn.Parameter(torch.zeros(1, 4, d_model)) # Sequence length bounded to L = 4
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=n_heads, 
            dim_feedforward=d_ff,
            norm_first=True, # Pre-LN configuration to explicitly bound variance
            batch_first=True,
            activation='gelu'
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.lm_head = nn.Linear(d_model, vocab_size)
        
        # Strict Xavier Topological Conditioning
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x):
        x = self.embedding(x) + self.pos_encoding
        # Explicit upper triangular causal mask to prevent acausal information leakage
        mask = nn.Transformer.generate_square_subsequent_mask(x.size(1)).to(x.device)
        out = self.transformer(x, mask=mask, is_causal=True)
        return self.lm_head(out)

# ==========================================
# 3. SPECTRAL EXTRACTION (HVP + POWER ITERATION)
# ==========================================
def compute_lambda_max(model, loss_fn, x, y, num_iterations=20):
    """
    Iterative approximation of the dominant eigenvalue (\lambda_max) 
    using Hessian-Vector Products.
    """
    # Explicitly disable fused attention kernels (FlashAttention) 
    # to unroll the exact computational graph for double backpropagation
    with torch.backends.cuda.sdp_kernel(enable_flash=False, enable_math=True, enable_mem_efficient=False):
        model.zero_grad()
        outputs = model(x)
        
        # Evaluated exclusively on the final sequence token
        loss = loss_fn(outputs[:, -1, :], y[:, -1]) 
        
        params = [p for p in model.parameters() if p.requires_grad]
        
        # First-order derivative (Gradient)
        grads = torch.autograd.grad(loss, params, create_graph=True)
        
        # Initialize random vector v0
        v = [torch.randn_like(p) for p in params]
        v_norm = torch.sqrt(sum((x**2).sum() for x in v))
        v = [x / v_norm for x in v]
        
        # Power Iteration numerical method
        for _ in range(num_iterations):
            grad_v = sum((g * v_i).sum() for g, v_i in zip(grads, v))
            hvp = torch.autograd.grad(grad_v, params, retain_graph=True)
            
            hvp_norm = torch.sqrt(sum((x**2).sum() for x in hvp))
            v = [x / (hvp_norm + 1e-8) for x in hvp] # L2 Normalization to prevent numerical overflow
            
        # Rayleigh Quotient evaluation
        grad_v = sum((g * v_i).sum() for g, v_i in zip(grads, v))
        final_hvp = torch.autograd.grad(grad_v, params)
        lambda_max = sum((v_i * h_i).sum() for v_i, h_i in zip(v, final_hvp))
        
        return lambda_max.item()

# ==========================================
# 3.5 WEIGHT NORM INSTRUMENTATION
# ==========================================
def compute_weight_l2_norm(model):
    """
    Computes the total L2 norm across all trainable parameters:
        ||theta||_2 = sqrt(sum_i theta_i^2)
    Tracked to test whether Adam and SGD (both using PyTorch's coupled
    weight_decay=1.0) converge to a similarly small-norm region regardless
    of the base optimizer, which would explain why their lambda_max and
    accuracy trajectories are nearly indistinguishable from each other.
    """
    total_sq = 0.0
    for p in model.parameters():
        if p.requires_grad:
            total_sq += torch.sum(p.detach() ** 2).item()
    return total_sq ** 0.5


# ==========================================
# 4. PRIMARY OPTIMIZATION HORIZON
# ==========================================
def train_model(optimizer_name, device):
    print(f"\n--- Initiating Topological Ablation: {optimizer_name} ---")
    
    # Data manifold preparation
    train_dataset = ModularAdditionDataset(split='train')
    val_dataset = ModularAdditionDataset(split='val')
    # Static batch size of B=256
    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False)
    
    # Reset seed to guarantee strictly identical parametric initializations
    torch.manual_seed(42)
    model = SmallCausalTransformer().to(device)
    loss_fn = nn.CrossEntropyLoss()
    
    # Optimizer Selection (Coupled vs. Decoupled)
    lr = 1e-3
    weight_decay = 1.0 # Topological catalyst for geometric compression
    
    if optimizer_name == "AdamW":
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay, betas=(0.9, 0.98))
    elif optimizer_name == "Adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay, betas=(0.9, 0.98))
    elif optimizer_name == "SGD":
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay, momentum=0.9)

    steps = 25000 # Extended asymptotic optimization horizon
    eval_interval = 500
    
    history = {'step': [], 'lambda_max': [], 'val_acc': [], 'weight_l2_norm': []}
    step_iterator = iter(train_loader)
    
    for step in range(1, steps + 1):
        try:
            x, y = next(step_iterator)
        except StopIteration:
            step_iterator = iter(train_loader)
            x, y = next(step_iterator)
            
        x, y = x.to(device), y.to(device)
        
        model.train()
        optimizer.zero_grad()
        outputs = model(x)
        loss = loss_fn(outputs[:, -1, :], y[:, -1])
        loss.backward()
        optimizer.step()
        
        # Periodic Spectral Evaluation (Active Topological Monitoring)
        if step % eval_interval == 0 or step == 1:
            # 1. Compute Predictive Accuracy
            model.eval()
            correct, total = 0, 0
            with torch.no_grad():
                for vx, vy in val_loader:
                    vx, vy = vx.to(device), vy.to(device)
                    v_out = model(vx)
                    preds = torch.argmax(v_out[:, -1, :], dim=-1)
                    correct += (preds == vy[:, -1]).sum().item()
                    total += vy.size(0)
            val_acc = correct / total
            
            # 2. Extract Dominant Eigenvalue (\lambda_max)
            model.train() # Requires active gradient graph
            l_max = compute_lambda_max(model, loss_fn, x, y)

            # 3. Compute total L2 norm of trainable weights (diagnostic)
            w_norm = compute_weight_l2_norm(model)
            
            history['step'].append(step)
            history['val_acc'].append(val_acc)
            history['lambda_max'].append(l_max)
            history['weight_l2_norm'].append(w_norm)
            print(f"Step {step:05d} | Val Acc: {val_acc:.4f} | Lambda_Max: {l_max:.2f} | Weight L2 Norm: {w_norm:.2f}")
            
    return history

# ==========================================
# 5. EXECUTION AND ANALYTICAL VISUALIZATION
# ==========================================
if __name__ == "__main__":
    import json

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Executing on hardware architecture: {device}")
    
    optimizers = ["AdamW", "Adam", "SGD"]
    results = {}
    
    for opt in optimizers:
        results[opt] = train_model(opt, device)

    # Persist raw results to disk BEFORE plotting, so the numbers can be
    # verified directly rather than only inspected visually in the figure.
    with open("optimizer_ablation_results.json", "w") as f:
        json.dump(results, f)
    print("Raw results saved to optimizer_ablation_results.json")
        
    # Generate and save analytical figure for Appendix A
    plt.figure(figsize=(12, 8))
    
    # Subplot 1: Validation Accuracy Trajectory
    plt.subplot(2, 1, 1)
    for opt in optimizers:
        plt.plot(results[opt]['step'], results[opt]['val_acc'], label=f'{opt} (Val Acc)')
    plt.title('Ablation Study: Algorithmic Generalization (Accuracy)')
    plt.ylabel('Validation Accuracy')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Subplot 2: Spectral Evolution (Hessian Lambda Max)
    plt.subplot(2, 1, 2)
    for opt in optimizers:
        plt.plot(results[opt]['step'], results[opt]['lambda_max'], label=f'{opt} (\u03bb_max)', alpha=0.8)
    plt.title('Spectral Dynamics: Impact of Coupled vs. Decoupled Weight Decay')
    plt.xlabel('Optimization Steps')
    plt.ylabel('Dominant Eigenvalue (\u03bb_max)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('optimizer_ablation_results.png', dpi=300)
    print("\nTopological ablation successfully concluded. Analytical figure saved as 'optimizer_ablation_results.png'.")
