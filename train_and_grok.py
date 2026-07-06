"""
train_and_grok.py

Executes the extended asymptotic optimization trajectory to induce 
delayed algorithmic generalization (the topological phase transition).
This script natively supports topological ablation studies by actively 
manipulating the decoupled weight decay parameter. It automatically saves 
parametric states (checkpoints) and macroscopic telemetry (loss, accuracy) 
strictly required for the subsequent continuous Hessian spectral analysis.
"""

import os
import json
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# Import custom architectural modules
from generate_dataset import ModularArithmeticDataset
from model_architecture import ToyTransformer

# =====================================================================
# 🔬 MACROSCOPIC HYPERPARAMETER CONFIGURATION & TOPOLOGICAL ABLATION
# =====================================================================
parser = argparse.ArgumentParser(description="Train ToyTransformer on modular addition and track grokking telemetry.")
parser.add_argument("--seed", type=int, default=42, help="Random seed for dataset split, init, and shuffling.")
parser.add_argument("--train_ratio", type=float, default=0.5, help="Fraction of the total data manifold used for training (0.5 / 0.25 / 0.10).")
parser.add_argument("--weight_decay", type=float, default=1.0, help="Decoupled weight decay (AdamW). 0.0 disables the topological catalyst.")
parser.add_argument("--max_steps", type=int, default=25000, help="Optimization horizon.")
parser.add_argument("--run_name", type=str, default=None, help="Identifier for this run; used to namespace checkpoints/telemetry. Defaults to an auto-generated name from seed/ratio.")
args = parser.parse_args()

P_MODULO = 97
BATCH_SIZE = 256
LEARNING_RATE = 1e-3
MAX_STEPS = args.max_steps   # Optimization horizon (strictly required for the asymptotic regime)
EVAL_EVERY = 100    # Extract macroscopic telemetry and save parametric states every N steps
SEED = args.seed

# --- ABLATION STUDY: DECOUPLED WEIGHT DECAY ---
# 1.0 -> Catalyst for delayed generalization (Actively forces geometric circuit compression)
# 0.0 -> Topological Ablation (Architecture densely memorizes but permanently fails to traverse the geometric barrier)
WEIGHT_DECAY = args.weight_decay
TRAIN_RATIO = args.train_ratio

RUN_NAME = args.run_name or f"ratio{int(TRAIN_RATIO*100)}_seed{SEED}"
CHECKPOINT_DIR = os.path.join("checkpoints", RUN_NAME)
TELEMETRY_FILE = f"grokking_telemetry_{RUN_NAME}.json"
# =====================================================================


def train_model() -> None:
    """
    Primary optimization loop. Initializes the discrete data manifold, structural architecture, 
    and decoupled optimizer, executing the asymptotic trajectory over MAX_STEPS 
    while continuously recording macroscopic telemetry.
    """
    # 1. Hardware initialization
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Executing asymptotic optimization on hardware architecture: {device}")
    print(f"Run: {RUN_NAME} | Seed: {SEED} | Train Ratio: {TRAIN_RATIO} | "
          f"Weight Decay: {WEIGHT_DECAY} | Optimization Steps: {MAX_STEPS}")

    # Seed torch's global RNG up front so DataLoader shuffling, Xavier init, and
    # dropout are all reproducible per-seed while genuinely varying across seeds.
    torch.manual_seed(SEED)

    # 2. Data Manifold Instantiation
    # Utilizing drop_last=True ensures strictly uniform tensor dimensions for mathematically stable Hessian extraction
    train_dataset = ModularArithmeticDataset(p=P_MODULO, split='train', train_ratio=TRAIN_RATIO, seed=SEED)
    val_dataset = ModularArithmeticDataset(p=P_MODULO, split='val', train_ratio=TRAIN_RATIO, seed=SEED)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, drop_last=True)

    # 3. Architecture & Optimizer Instantiation
    model = ToyTransformer(vocab_size=train_dataset.vocab_size).to(device)
    
    # Utilizing the AdamW operator to decouple weight decay from momentum-based gradient updates,
    # exerting pure, directional geometric pressure on the parameter space.
    optimizer = optim.AdamW(
        model.parameters(), 
        lr=LEARNING_RATE, 
        weight_decay=WEIGHT_DECAY, 
        betas=(0.9, 0.98)
    )
    criterion = nn.CrossEntropyLoss()

    # 4. Macroscopic Telemetry and Parametric State Tracking setup
    telemetry = {
        "steps": [],
        "train_loss": [],
        "val_loss": [],
        "val_accuracy": []
    }
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    # 5. Continuous Optimization Trajectory
    model.train()
    step = 0
    train_iterator = iter(train_loader)

    print("\nInitiating continuous optimization trajectory...")
    print("This phase requires extended computation to reach the necessary asymptotic regime.")

    while step < MAX_STEPS:
        try:
            x_batch, y_batch = next(train_iterator)
        except StopIteration:
            train_iterator = iter(train_loader)
            x_batch, y_batch = next(train_iterator)
            
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)
        
        # Forward propagation
        optimizer.zero_grad()
        logits = model(x_batch)
        
        # We exclusively evaluate the predictive probability distribution on the final sequence token (the true mapping 'c')
        # logits shape: (batch_size, seq_len, vocab_size)
        final_logits = logits[:, -1, :]
        final_targets = y_batch[:, -1]
        
        loss = criterion(final_logits, final_targets)
        
        # Backpropagation and parametric update
        loss.backward()
        optimizer.step()
        
        # --- Structural Evaluation and Macroscopic Telemetry Logging ---
        if step % EVAL_EVERY == 0:
            model.eval()
            val_loss_total = 0.0
            correct_predictions = 0
            total_predictions = 0
            
            with torch.no_grad():
                for x_val, y_val in val_loader:
                    x_val, y_val = x_val.to(device), y_val.to(device)
                    v_logits = model(x_val)
                    v_final_logits = v_logits[:, -1, :]
                    v_final_targets = y_val[:, -1]
                    
                    v_loss = criterion(v_final_logits, v_final_targets)
                    val_loss_total += v_loss.item()
                    
                    # Calculate Predictive Accuracy
                    predictions = torch.argmax(v_final_logits, dim=-1)
                    correct_predictions += (predictions == v_final_targets).sum().item()
                    total_predictions += v_final_targets.size(0)
                    
            avg_val_loss = val_loss_total / len(val_loader)
            val_acc = correct_predictions / total_predictions
            
            # Persist macroscopic telemetry
            telemetry["steps"].append(step)
            telemetry["train_loss"].append(loss.item())
            telemetry["val_loss"].append(avg_val_loss)
            telemetry["val_accuracy"].append(val_acc)
            
            print(f"Step {step:05d} | In-Sample Risk (Loss): {loss.item():.4f} | "
                  f"Population Risk (Loss): {avg_val_loss:.4f} | Predictive Accuracy: {val_acc:.4f}")
            
            # Save parametric weights (Topological Checkpointing)
            # High-resolution state preservation during initial turbulence, spaced asymptotically to optimize storage
            if step < 2000 or step % (EVAL_EVERY * 5) == 0:
                torch.save(model.state_dict(), f"{CHECKPOINT_DIR}/model_step_{step}.pt")
                
            model.train()  # Revert to active gradient graph
            
        step += 1

    # 6. Persist final telemetry to disk (include run metadata for downstream aggregation)
    telemetry["run_name"] = RUN_NAME
    telemetry["seed"] = SEED
    telemetry["train_ratio"] = TRAIN_RATIO
    telemetry["weight_decay"] = WEIGHT_DECAY
    telemetry["checkpoint_dir"] = CHECKPOINT_DIR

    with open(TELEMETRY_FILE, "w") as f:
        json.dump(telemetry, f)

    print(f"\nAsymptotic optimization concluded! Macroscopic telemetry successfully saved to '{TELEMETRY_FILE}'.")


if __name__ == "__main__":
    train_model()
