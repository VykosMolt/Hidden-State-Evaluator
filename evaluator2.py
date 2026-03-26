import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionPool(nn.Module):
    """
    Learned attention pooling over the token dimension.
    Low-rank bottleneck: projects to a small attn_dim for scoring,
    preventing the pooling layer from memorizing token-specific features.
    Forces it to learn simple positional/structural weighting instead.

    Input: [batch, seq_len, hidden_dim], attention_mask [batch, seq_len]
    Output: [batch, hidden_dim]
    """

    def __init__(self, hidden_dim, attn_dim=128):
        super().__init__()
        self.proj = nn.Linear(hidden_dim, attn_dim, bias=False)
        self.query = nn.Parameter(torch.randn(attn_dim) * 0.01)

    def forward(self, hidden, attention_mask):
        # hidden: [batch, seq_len, hidden_dim]
        # attention_mask: [batch, seq_len]

        keys = self.proj(hidden)  # [batch, seq_len, attn_dim]
        scores = keys @ self.query  # [batch, seq_len]

        scores = scores.masked_fill(attention_mask == 0, float("-inf"))
        weights = F.softmax(scores, dim=-1)  # [batch, seq_len]

        pooled = torch.einsum("bs,bsd->bd", weights, hidden)  # [batch, hidden_dim]
        return pooled


class ConstitutionalEvaluatorV2(nn.Module):
    """
    Constitutional Evaluator V2 — GRU-based temporal model over loop states
    with learned attention pooling.

    Now receives raw hidden states [batch, seq_len, hidden_dim] per loop step
    and learns which token positions carry preference signal, rather than
    relying on uniform mean pooling.

    Input: list of [batch, seq_len, hidden_dim] tensors + attention_mask
    Output: scalar alignment score (unbounded, pairwise ranking loss)
    """

    def __init__(self, hidden_dim=2048, gru_hidden=512, gru_layers=2,
                 intermediate_size=256, dropout=0.1):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.gru_hidden = gru_hidden

        # learned pooling replaces mean_pool
        self.attention_pool = AttentionPool(hidden_dim)

        self.input_norm = nn.LayerNorm(hidden_dim)
        self.input_proj = nn.Linear(hidden_dim, gru_hidden)

        self.gru = nn.GRU(
            input_size=gru_hidden,
            hidden_size=gru_hidden,
            num_layers=gru_layers,
            batch_first=False,
            dropout=dropout if gru_layers > 1 else 0.0,
            bidirectional=False
        )

        scorer_input_dim = gru_hidden * 2

        self.scorer = nn.Sequential(
            nn.LayerNorm(scorer_input_dim),
            nn.Linear(scorer_input_dim, intermediate_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(intermediate_size, 1)
        )

    def _pool_and_prepare(self, hidden_states_list, attention_mask):
        """Pool raw hidden states, normalize, project, and stack."""
        pooled = []
        for h in hidden_states_list:
            p = self.attention_pool(h, attention_mask)  # [batch, hidden_dim]
            pooled.append(p)

        normed = [self.input_norm(p) for p in pooled]
        projected = [self.input_proj(n) for n in normed]
        return torch.stack(projected, dim=0), projected[-1]

    def forward(self, hidden_states_list, attention_mask):
        """
        Full trajectory forward pass.
        Args:
            hidden_states_list: list of [batch, seq_len, hidden_dim] tensors
            attention_mask: [batch, seq_len]
        Returns:
            score: [batch, 1]
        """
        seq, final_proj = self._pool_and_prepare(hidden_states_list, attention_mask)

        _, final_hidden = self.gru(seq)
        gru_out = final_hidden[-1]  # [batch, gru_hidden]

        combined = torch.cat([gru_out, final_proj], dim=-1)
        return self.scorer(combined)

    def trajectory(self, hidden_states_list, attention_mask):
        """
        Score at each loop step by running GRU incrementally.
        Args:
            hidden_states_list: list of [batch, seq_len, hidden_dim] tensors
            attention_mask: [batch, seq_len]
        Returns:
            scores: list of [batch, 1] tensors
            trajectory: list of scalar floats
        """
        pooled = [self.attention_pool(h, attention_mask) for h in hidden_states_list]
        normed = [self.input_norm(p) for p in pooled]
        projected = [self.input_proj(n) for n in normed]

        scores = []
        hidden = None

        for step_proj in projected:
            out, hidden = self.gru(step_proj.unsqueeze(0), hidden)
            gru_out = hidden[-1]
            combined = torch.cat([gru_out, step_proj], dim=-1)
            scores.append(self.scorer(combined))

        trajectory = [s.mean().item() for s in scores]
        return scores, trajectory


def mean_pool(hidden, attention_mask):
    """Mask-weighted mean pooling over token dimension. Kept for linear probe."""
    mask = attention_mask.unsqueeze(-1).float()
    return (hidden * mask).sum(dim=1) / mask.sum(dim=1)


def validate_hook_output(output):
    """
    Validates that the hook captured the expected structure from Ouro.
    Raises AssertionError with a diagnostic message if the format changed.
    """
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


def linear_probe_test(pooled_chosen_list, pooled_rejected_list, use_final_only=True):
    """
    Pairwise linear probe using mean-pooled states.
    Still uses mean pooling since this is a diagnostic, not the trained model.
    """
    from sklearn.linear_model import LogisticRegression
    import numpy as np

    features = []
    labels = []

    for chosen_states, rejected_states in zip(pooled_chosen_list, pooled_rejected_list):
        if use_final_only:
            c = chosen_states[-1].detach().cpu()
            r = rejected_states[-1].detach().cpu()
        else:
            c = torch.cat(chosen_states, dim=-1).detach().cpu()
            r = torch.cat(rejected_states, dim=-1).detach().cpu()

        diff = c - r
        for row in diff.numpy():
            features.append(row)
            labels.append(1)
        for row in (-diff).numpy():
            features.append(row)
            labels.append(0)

    X = np.array(features)
    y = np.array(labels)

    n = len(y)
    split = int(0.8 * n)
    idx = np.random.RandomState(42).permutation(n)
    X_train, X_test = X[idx[:split]], X[idx[split:]]
    y_train, y_test = y[idx[:split]], y[idx[split:]]

    clf = LogisticRegression(max_iter=1000, C=1.0)
    clf.fit(X_train, y_train)
    acc = clf.score(X_test, y_test)

    print(f"Linear probe accuracy: {acc:.4f} (final_only={use_final_only})")
    print(f"  Train samples: {len(y_train)}, Test samples: {len(y_test)}")
    return acc


def test_evaluator_v2():
    """Sanity checks for ConstitutionalEvaluatorV2."""
    evaluator = ConstitutionalEvaluatorV2()

    batch, seq_len, hidden_dim = 2, 128, 2048
    mask = torch.ones(batch, seq_len)

    # forward pass
    dummy_states = [torch.randn(batch, seq_len, hidden_dim) for _ in range(4)]
    score = evaluator(dummy_states, mask)
    print(f"Forward pass — Output: {score.shape}")
    assert score.shape == (batch, 1)
    print("Forward pass: OK")

    # trajectory mode
    scores, trajectory = evaluator.trajectory(dummy_states, mask)
    print(f"Trajectory — {len(scores)} scores across 4 loop steps")
    print(f"Trajectory values: {[f'{t:.4f}' for t in trajectory]}")
    assert len(scores) == 4
    print("Trajectory mode: OK")

    assert not torch.allclose(scores[0], scores[-1]), \
        "Trajectory scores should differ across steps"
    print("Trajectory dynamics: OK")

    # variable loop counts
    for n_steps in [1, 2, 6, 8]:
        states = [torch.randn(batch, seq_len, hidden_dim) for _ in range(n_steps)]
        s = evaluator(states, mask)
        assert s.shape == (batch, 1), f"Failed for {n_steps} steps"
    print("Variable loop counts (1, 2, 6, 8): OK")

    # attention pooling with actual padding
    mask_padded = torch.ones(batch, seq_len)
    mask_padded[:, 64:] = 0  # half padded
    score_padded = evaluator(dummy_states, mask_padded)
    assert score_padded.shape == (batch, 1)
    print("Padded attention pooling: OK")

    # validate hook output
    validate_hook_output(dummy_states)
    print("Hook validation: OK")

    print("\nAll checks passed.")


if __name__ == "__main__":
    test_evaluator_v2()
