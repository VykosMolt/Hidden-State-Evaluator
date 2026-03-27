import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionPool(nn.Module):
    """
    Low-rank learned attention pooling over token dimension.
    attn_dim=256 (up from 128 — gives more capacity without the
    full-rank collapse we saw at 2048).
    """

    def __init__(self, hidden_dim, attn_dim=256):
        super().__init__()
        self.proj = nn.Linear(hidden_dim, attn_dim, bias=False)
        self.query = nn.Parameter(torch.randn(attn_dim) * 0.01)

    def forward(self, hidden, attention_mask):
        keys = self.proj(hidden)  # [batch, seq_len, attn_dim]
        scores = keys @ self.query  # [batch, seq_len]
        scores = scores.masked_fill(attention_mask == 0, float("-inf"))
        weights = F.softmax(scores, dim=-1)
        pooled = torch.einsum("bs,bsd->bd", weights, hidden)
        return pooled


class ConstitutionalEvaluatorTest(nn.Module):
    """
    Minimal evaluator — no GRU, no temporal modeling.
    Attention-pools the FINAL loop state only, then scores via MLP.

    This is the closest architecture to the linear probe that got 93.75%.
    The hypothesis: sequential bottlenecks (pool -> proj -> GRU -> scorer)
    are destroying signal. A direct pool -> score path should recover more.

    Input: list of [batch, seq_len, hidden_dim] tensors + attention_mask
    Output: scalar alignment score [batch, 1]
    """

    def __init__(self, hidden_dim=2048, intermediate_size=512, dropout=0.1):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.attention_pool = AttentionPool(hidden_dim)
        self.input_norm = nn.LayerNorm(hidden_dim)

        # wider MLP since it's the only nonlinear transform
        self.scorer = nn.Sequential(
            nn.Linear(hidden_dim, intermediate_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(intermediate_size, intermediate_size // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(intermediate_size // 2, 1)
        )

    def forward(self, hidden_states_list, attention_mask):
        """
        Score using only the final loop state.
        Args:
            hidden_states_list: list of [batch, seq_len, hidden_dim] tensors
            attention_mask: [batch, seq_len]
        Returns:
            score: [batch, 1]
        """
        final_state = hidden_states_list[-1]  # [batch, seq_len, hidden_dim]
        pooled = self.attention_pool(final_state, attention_mask)  # [batch, hidden_dim]
        normed = self.input_norm(pooled)  # [batch, hidden_dim]
        return self.scorer(normed)


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


def test_evaluator():
    evaluator = ConstitutionalEvaluatorTest()

    batch, seq_len, hidden_dim = 2, 128, 2048
    mask = torch.ones(batch, seq_len)

    # forward pass with 4 loop states (only last used)
    dummy_states = [torch.randn(batch, seq_len, hidden_dim) for _ in range(4)]
    score = evaluator(dummy_states, mask)
    print(f"Forward pass — Output: {score.shape}")
    assert score.shape == (batch, 1)
    print("Forward pass: OK")

    # verify only final state matters
    dummy_states_2 = [torch.randn(batch, seq_len, hidden_dim) for _ in range(3)]
    dummy_states_2.append(dummy_states[-1])  # same final state
    score_2 = evaluator(dummy_states_2, mask)
    assert torch.allclose(score, score_2), "Scores should match when final state matches"
    print("Final-state-only: OK")

    # variable loop counts
    for n_steps in [1, 2, 6, 8]:
        states = [torch.randn(batch, seq_len, hidden_dim) for _ in range(n_steps)]
        s = evaluator(states, mask)
        assert s.shape == (batch, 1), f"Failed for {n_steps} steps"
    print("Variable loop counts: OK")

    # padded input
    mask_padded = torch.ones(batch, seq_len)
    mask_padded[:, 64:] = 0
    score_padded = evaluator(dummy_states, mask_padded)
    assert score_padded.shape == (batch, 1)
    print("Padded attention pooling: OK")

    # param count
    total_params = sum(p.numel() for p in evaluator.parameters())
    trainable_params = sum(p.numel() for p in evaluator.parameters() if p.requires_grad)
    print(f"Total params: {total_params:,}")
    print(f"Trainable params: {trainable_params:,}")

    print("\nAll checks passed.")


if __name__ == "__main__":
    test_evaluator()
