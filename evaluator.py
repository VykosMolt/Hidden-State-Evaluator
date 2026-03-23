import torch
import torch.nn as nn


class ConstitutionalEvaluator(nn.Module):
    """
    Constitutional Evaluator — the "amygdala" component of the CLT architecture.

    Two modes:
    1. Single score: concatenated last N loop hidden states → scalar
    2. Trajectory: sliding window over all loop steps → score per step

    Input (single): [batch, hidden_dim * n_concat]
    Output: scalar alignment score (unbounded, use with pairwise ranking loss)
    """

    def __init__(self, hidden_dim=2048, n_concat=3, intermediate_size=1024):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.n_concat = n_concat
        concat_dim = hidden_dim * n_concat

        self.network = nn.Sequential(
            nn.LayerNorm(concat_dim),
            nn.Linear(concat_dim, intermediate_size),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(intermediate_size, 256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, 1)
            # no sigmoid — pairwise ranking loss handles score magnitude
        )

    def forward(self, pooled_hidden):
        """
        Single score mode.
        Args:
            pooled_hidden: [batch, hidden_dim * n_concat]
        Returns:
            score: [batch, 1]
        """
        return self.network(pooled_hidden)

    def trajectory(self, hidden_states_list):
        """
        Trajectory mode — sliding window over loop iterations.
        Each step sees its own state plus n_concat-1 previous states.
        Args:
            hidden_states_list: list of [batch, hidden_dim] pooled tensors
        Returns:
            scores: list of [batch, 1] tensors
            trajectory: list of scalar floats for monitoring
        """
        scores = []

        for i in range(len(hidden_states_list)):
            # sliding window of last n_concat states
            window = hidden_states_list[max(0, i - self.n_concat + 1): i + 1]

            # pad with first state if window not full yet
            while len(window) < self.n_concat:
                window = [torch.zeros_like(window[0])] + window

            combined = torch.cat(window, dim=-1)  # [batch, hidden_dim * n_concat]
            scores.append(self.network(combined))

        trajectory = [s.mean().item() for s in scores]
        return scores, trajectory


def mean_pool(hidden, attention_mask):
    """Mask-weighted mean pooling over token dimension."""
    mask = attention_mask.unsqueeze(-1).float()
    return (hidden * mask).sum(dim=1) / mask.sum(dim=1)


def concat_loop_states(hidden_states_list, n_concat=3):
    """
    Concatenates the last n_concat pooled loop states.
    Args:
        hidden_states_list: list of [batch, hidden_dim] pooled tensors
    Returns:
        combined: [batch, hidden_dim * n_concat]
    """
    assert hidden_states_list[0].dim() == 2, "Expected pooled hidden states [batch, hidden_dim]"

    selected = hidden_states_list[-n_concat:]

    while len(selected) < n_concat:
        selected = [torch.zeros_like(selected[0])] + selected

    return torch.cat(selected, dim=-1)


def test_evaluator():
    """Sanity checks for both modes."""
    evaluator = ConstitutionalEvaluator()

    # single score mode
    dummy_input = torch.randn(2, 2048 * 3)
    score = evaluator(dummy_input)
    print(f"Single score — Input: {dummy_input.shape} → Output: {score.shape}")
    assert score.shape == (2, 1)
    print("Single score mode: OK")

    # trajectory mode
    dummy_loop_states = [torch.randn(2, 2048) for _ in range(4)]
    scores, trajectory = evaluator.trajectory(dummy_loop_states)
    print(f"Trajectory — {len(scores)} scores across {len(dummy_loop_states)} loop steps")
    print(f"Trajectory values: {[f'{t:.4f}' for t in trajectory]}")

    # verify sliding window is actually different per step
    assert not torch.allclose(scores[0], scores[-1]), "Trajectory scores should differ across steps"
    print("Trajectory sliding window: OK")

    print("\nAll checks passed.")


if __name__ == "__main__":
    test_evaluator()
