import torch
import torch.nn as nn


class ConstitutionalEvaluatorV2(nn.Module):
    """
    Constitutional Evaluator V2 — GRU-based temporal model over loop states.

    Instead of concatenating a sliding window of hidden states, this version
    processes the sequence of loop hidden states through a GRU, capturing how
    the representation evolves across iterations rather than just its endpoint.

    The GRU hidden state after processing all loop steps encodes the full
    trajectory dynamics — early uncertainty, mid-loop refinement, and final
    convergence — as a single fixed-size vector fed into the scoring head.

    Input: list of [batch, hidden_dim] pooled tensors (one per loop step)
    Output: scalar alignment score (unbounded, use with pairwise ranking loss)
    """

    def __init__(self, hidden_dim=2048, gru_hidden=512, intermediate_size=256):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.gru_hidden = gru_hidden

        # instance norm applied per hidden state before GRU
        # removes absolute scale differences between examples
        self.input_norm = nn.LayerNorm(hidden_dim)

        # GRU processes the sequence of loop hidden states
        # input: [seq_len, batch, hidden_dim]
        # output hidden: [batch, gru_hidden]
        self.gru = nn.GRU(
            input_size=hidden_dim,
            hidden_size=gru_hidden,
            num_layers=1,
            batch_first=False,
            bidirectional=False
        )

        # scoring head takes final GRU hidden state
        self.scorer = nn.Sequential(
            nn.LayerNorm(gru_hidden),
            nn.Linear(gru_hidden, intermediate_size),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(intermediate_size, 1)
            # no sigmoid — pairwise ranking loss handles magnitude
        )

    def forward(self, hidden_states_list):
        """
        Full trajectory mode — GRU over all loop steps.
        Args:
            hidden_states_list: list of [batch, hidden_dim] pooled tensors
        Returns:
            score: [batch, 1] final alignment score
        """
        # normalize each step independently
        normed = [self.input_norm(h) for h in hidden_states_list]

        # stack to [seq_len, batch, hidden_dim]
        seq = torch.stack(normed, dim=0)

        # run GRU — we only need the final hidden state
        _, final_hidden = self.gru(seq)  # final_hidden: [1, batch, gru_hidden]
        final_hidden = final_hidden.squeeze(0)  # [batch, gru_hidden]

        return self.scorer(final_hidden)

    def trajectory(self, hidden_states_list):
        """
        Trajectory mode — score at each loop step by running GRU up to that point.
        Useful for monitoring how the alignment score evolves during inference.
        Args:
            hidden_states_list: list of [batch, hidden_dim] pooled tensors
        Returns:
            scores: list of [batch, 1] tensors, one per loop step
            trajectory: list of scalar floats for monitoring
        """
        normed = [self.input_norm(h) for h in hidden_states_list]

        scores = []
        hidden = None  # GRU hidden state carries forward

        for step_input in normed:
            # step_input: [batch, hidden_dim] → unsqueeze to [1, batch, hidden_dim]
            out, hidden = self.gru(step_input.unsqueeze(0), hidden)
            # hidden: [1, batch, gru_hidden]
            step_score = self.scorer(hidden.squeeze(0))  # [batch, 1]
            scores.append(step_score)

        trajectory = [s.mean().item() for s in scores]
        return scores, trajectory


def mean_pool(hidden, attention_mask):
    """Mask-weighted mean pooling over token dimension."""
    mask = attention_mask.unsqueeze(-1).float()
    return (hidden * mask).sum(dim=1) / mask.sum(dim=1)


def test_evaluator_v2():
    """Sanity checks for ConstitutionalEvaluatorV2."""
    evaluator = ConstitutionalEvaluatorV2()

    # full trajectory forward pass
    dummy_loop_states = [torch.randn(2, 2048) for _ in range(4)]
    score = evaluator(dummy_loop_states)
    print(f"Forward pass — Output: {score.shape}")
    assert score.shape == (2, 1)
    print("Forward pass: OK")

    # trajectory mode
    scores, trajectory = evaluator.trajectory(dummy_loop_states)
    print(f"Trajectory — {len(scores)} scores across {len(dummy_loop_states)} loop steps")
    print(f"Trajectory values: {[f'{t:.4f}' for t in trajectory]}")
    assert len(scores) == 4
    print("Trajectory mode: OK")

    # verify trajectory scores differ across steps (GRU sees different context)
    assert not torch.allclose(scores[0], scores[-1]), "Trajectory scores should differ across steps"
    print("Trajectory dynamics: OK")

    # verify variable loop counts work
    short_loop = [torch.randn(2, 2048) for _ in range(2)]
    score_short = evaluator(short_loop)
    assert score_short.shape == (2, 1)
    print("Variable loop count: OK")

    print("\nAll checks passed.")


if __name__ == "__main__":
    test_evaluator_v2()
