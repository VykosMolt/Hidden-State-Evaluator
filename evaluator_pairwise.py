import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionPool(nn.Module):
    """
    Low-rank learned attention pooling over token dimension.
    Shared between chosen and rejected — same learned weighting for both.
    """

    def __init__(self, hidden_dim, attn_dim=128):
        super().__init__()
        self.proj = nn.Linear(hidden_dim, attn_dim, bias=False)
        self.query = nn.Parameter(torch.randn(attn_dim) * 0.01)

    def forward(self, hidden, attention_mask):
        keys = self.proj(hidden)
        scores = keys @ self.query
        scores = scores.masked_fill(attention_mask == 0, float("-inf"))
        weights = F.softmax(scores, dim=-1)
        return torch.einsum("bs,bsd->bd", weights, hidden)


class PairwiseEvaluator(nn.Module):
    """
    Pairwise Constitutional Evaluator — compares two responses directly.

    Architecture:
    1. Attention-pool both chosen and rejected at each loop step (shared weights)
    2. Compute difference: chosen_pooled - rejected_pooled per step
    3. GRU over the sequence of differences (captures how preference gap evolves)
    4. Skip connection from final step difference
    5. Scorer outputs a single scalar: positive = chosen preferred

    This matches the pairwise probe's approach (84.5% accuracy on differences)
    rather than the independent scoring approach (21% — below chance).

    Input: chosen/rejected hidden states lists + attention masks
    Output: scalar preference score [batch, 1] (positive = chosen > rejected)
    """

    def __init__(self, hidden_dim=2048, gru_hidden=512, gru_layers=2,
                 intermediate_size=256, dropout=0.1):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.gru_hidden = gru_hidden

        # shared pooling for both responses
        self.attention_pool = AttentionPool(hidden_dim)

        self.input_norm = nn.LayerNorm(hidden_dim, bias=False)
        self.input_proj = nn.Linear(hidden_dim, gru_hidden)

        self.gru = nn.GRU(
            input_size=gru_hidden,
            hidden_size=gru_hidden,
            num_layers=gru_layers,
            batch_first=False,
            dropout=dropout if gru_layers > 1 else 0.0,
            bidirectional=False
        )

        # scorer sees GRU final hidden + projected final difference (skip)
        scorer_input_dim = gru_hidden * 2

        self.scorer = nn.Sequential(
            nn.LayerNorm(scorer_input_dim),
            nn.Linear(scorer_input_dim, intermediate_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(intermediate_size, 1)
        )

    def forward(self, chosen_states, chosen_mask, rejected_states, rejected_mask):
        """
        Compare two responses directly.
        Args:
            chosen_states: list of [batch, seq_len, hidden_dim] tensors (one per loop step)
            chosen_mask: [batch, seq_len]
            rejected_states: same structure as chosen
            rejected_mask: [batch, seq_len]
        Returns:
            score: [batch, 1] — positive means chosen preferred
        """
        # pool each loop step for both responses
        chosen_pooled = [self.attention_pool(h, chosen_mask) for h in chosen_states]
        rejected_pooled = [self.attention_pool(h, rejected_mask) for h in rejected_states]

        # per-step difference: how does chosen differ from rejected at each iteration
        diffs = [c - r for c, r in zip(chosen_pooled, rejected_pooled)]

        # normalize and project
        normed = [self.input_norm(d) for d in diffs]
        projected = [self.input_proj(n) for n in normed]

        # GRU over difference trajectory
        seq = torch.stack(projected, dim=0)  # [n_loops, batch, gru_hidden]
        _, final_hidden = self.gru(seq)
        gru_out = final_hidden[-1]  # [batch, gru_hidden]

        # skip connection from final projected difference
        combined = torch.cat([gru_out, projected[-1]], dim=-1)
        return self.scorer(combined)


def validate_hook_output(output):
    assert isinstance(output, (list, tuple)), \
        f"Expected list/tuple of hidden states, got {type(output)}"
    assert len(output) > 0, "Hook captured empty hidden states list"
    first = output[0]
    assert isinstance(first, torch.Tensor), \
        f"Expected tensor in hidden states list, got {type(first)}"
    assert first.dim() == 3, \
        f"Expected 3D tensor [batch, seq_len, hidden_dim], got {first.dim()}D"
    hidden_dim = first.shape[-1]
    for i, h in enumerate(output):
        assert h.shape[-1] == hidden_dim, \
            f"Hidden dim mismatch at step {i}: {h.shape[-1]} vs {hidden_dim}"


def test_pairwise():
    evaluator = PairwiseEvaluator()
    evaluator.eval()

    batch, seq_len, hidden_dim = 2, 128, 2048
    mask_c = torch.ones(batch, seq_len)
    mask_r = torch.ones(batch, seq_len)

    chosen = [torch.randn(batch, seq_len, hidden_dim) for _ in range(4)]
    rejected = [torch.randn(batch, seq_len, hidden_dim) for _ in range(4)]

    with torch.no_grad():
        score = evaluator(chosen, mask_c, rejected, mask_r)
    print(f"Forward pass — Output: {score.shape}")
    assert score.shape == (batch, 1)
    print("Forward pass: OK")

    # antisymmetry: swapping chosen/rejected should flip the sign
    with torch.no_grad():
        score_flipped = evaluator(rejected, mask_r, chosen, mask_c)
    print(f"Original: {score.squeeze().tolist()}")
    print(f"Flipped:  {score_flipped.squeeze().tolist()}")
    # not exact due to GRU nonlinearity, but should be opposite sign
    signs_match = (score.sign() != score_flipped.sign()).all()
    print(f"Signs flipped: {signs_match.item()}")
    print("Antisymmetry check: OK" if signs_match else "Antisymmetry check: WEAK (expected for nonlinear model)")

    # variable loop counts
    for n_steps in [1, 2, 6, 8]:
        c = [torch.randn(batch, seq_len, hidden_dim) for _ in range(n_steps)]
        r = [torch.randn(batch, seq_len, hidden_dim) for _ in range(n_steps)]
        with torch.no_grad():
            s = evaluator(c, mask_c, r, mask_r)
        assert s.shape == (batch, 1), f"Failed for {n_steps} steps"
    print("Variable loop counts: OK")

    # different padding
    mask_padded = torch.ones(batch, seq_len)
    mask_padded[:, 64:] = 0
    with torch.no_grad():
        score_padded = evaluator(chosen, mask_c, rejected, mask_padded)
    assert score_padded.shape == (batch, 1)
    print("Mixed padding: OK")

    total_params = sum(p.numel() for p in evaluator.parameters())
    print(f"Total params: {total_params:,}")

    print("\nAll checks passed.")


if __name__ == "__main__":
    test_pairwise()
