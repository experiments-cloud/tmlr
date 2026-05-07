"""
model_architecture.py

Defines a minimalistic causal Transformer (Decoder-only) explicitly optimized 
for delayed algorithmic generalization experiments and continuous Hessian 
spectral analysis.

The architectural specifications (d_model, n_heads, n_layers) strictly mirror
Table 1 of the paper. This mathematically bounded parameter scale allows for 
the continuous iterative resolution of the Hessian matrix without triggering 
spatial memory (OOM) bottlenecks during double backpropagation.
"""

import torch
import torch.nn as nn


class ToyTransformer(nn.Module):
    """
    A controlled-scale, autoregressive Transformer architecture.
    Dimensions are strictly bounded (d_model=128, 2 layers) to guarantee the 
    mathematical feasibility of extracting continuous Hessian topologies.
    """
    
    def __init__(self, 
                 vocab_size: int, 
                 d_model: int = 128, 
                 n_heads: int = 4, 
                 n_layers: int = 2, 
                 max_seq_len: int = 4, 
                 dropout: float = 0.1):
        """
        Initializes the architectural structure.
        
        Args:
            vocab_size (int): Dimension of the discrete vocabulary space (p + 3).
            d_model (int): Dimensionality of the latent manifold (Table 1).
            n_heads (int): Number of multi-head attention units (Table 1).
            n_layers (int): Number of structural Transformer blocks (Table 1).
            max_seq_len (int): Bounded sequence length (4 for algebraic [a, +, b, =]).
            dropout (float): Dropout probability.
        """
        super().__init__()
        self.d_model = d_model
        
        # 1. Manifold Encodings: Tokens and Absolute Spatial Positions
        # Since seq_len is strictly bounded (4), absolute spatial positions 
        # inject sufficient structural inductive bias.
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        
        # 2. Structural Transformer Backbone
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=n_heads, 
            dim_feedforward=d_model * 4, 
            dropout=dropout,
            batch_first=True,
            norm_first=True  # Pre-LN configuration to explicitly bound hidden state variance
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        
        # 3. Output Projection to Vocabulary Space
        self.lm_head = nn.Linear(d_model, vocab_size)
        
        # 4. Rigorous Topological Conditioning
        self._init_weights()

    def _init_weights(self) -> None:
        """
        Strict Xavier uniform parametric initialization. This formally rules out 
        geometric artifacts derived from scale invariance, establishing a 
        mathematically predictable starting point on the loss surface.
        """
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def generate_square_subsequent_mask(self, sz: int) -> torch.Tensor:
        """
        Generates an explicit upper triangular causal mask to mathematically 
        prevent acausal information leakage.
        
        Args:
            sz (int): Spatial sequence length.
            
        Returns:
            torch.Tensor: The upper triangular mask tensor.
        """
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Mathematical forward propagation of the autoregressive architecture.
        
        Args:
            x (torch.Tensor): Input sequence tensor of shape (batch_size, seq_len).
            
        Returns:
            torch.Tensor: Output logits mapped to the vocabulary dimension.
        """
        batch_size, seq_len = x.size()
        
        # Instantiate spatial positional indices dynamically
        positions = torch.arange(0, seq_len, dtype=torch.long, device=x.device)
        positions = positions.unsqueeze(0).expand(batch_size, seq_len)
        
        # Project into initial latent representation
        x_emb = self.token_emb(x) + self.pos_emb(positions)
        
        # Apply strict upper triangular bounding for causal learning
        causal_mask = self.generate_square_subsequent_mask(seq_len).to(x.device)
        
        # Traverse structural blocks
        out = self.transformer(x_emb, mask=causal_mask, is_causal=True)
        
        # Linear projection to the discrete algorithmic space
        logits = self.lm_head(out)
        
        return logits


# --- Independent Analytical Verification ---
if __name__ == "__main__":
    # Baseline analytical parameters (aligning with p=97 -> vocab_size=100)
    vocab_size = 100
    batch_size = 256
    seq_len = 4
    
    # Instantiate architecture
    model = ToyTransformer(vocab_size=vocab_size)
    
    # Generate synthetic input tensor matching optimal batch sizing
    dummy_input = torch.randint(0, vocab_size, (batch_size, seq_len))
    
    # Forward pass verification
    logits = model(dummy_input)
    
    # Parametric scale verification (Must approximate N ~ 422,000 as per Table 1)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print("\n--- Structural Architecture Telemetry ---")
    print(f"Total trainable parameters (N): {total_params:,}")
    print(f"Input tensor dimensions: {dummy_input.shape}")
    print(f"Output logits dimensions: {logits.shape} -> (Batch, Seq_Len, Vocab_Size)")
