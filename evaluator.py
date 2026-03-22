import torch
import torch.nn as nn

class ConstitutionalEvaluator(nn.Module):
    """
    Lightweight constitutional evaluator that takes Ouro's hidden states
    at each loop iteration and outputs an alignment score between 0 and 1.
    
    Architecture: mean pool hidden states → MLP → scalar alignment score
    Input: hidden state tensor of shape [batch, seq_len, 2048]
    Output: alignment score tensor of shape [batch, 1]
    """

    def __init__(self, hidden_size=2048, intermediate_size=512):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(hidden_size, intermediate_size),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(intermediate_size, 128),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, hidden_states):
        # hidden_states: [batch, seq_len, 2048]
        # mean pool across token dimension
        pooled = hidden_states.mean(dim=1)  # [batch, 2048]
        score = self.network(pooled)         # [batch, 1]
        return score


def test_evaluator():
    """Quick sanity check that the evaluator runs correctly."""
    evaluator = ConstitutionalEvaluator()
    
    # simulate a hidden state from Ouro
    dummy_hidden = torch.randn(1, 27, 2048)
    score = evaluator(dummy_hidden)
    
    print(f"Input shape: {dummy_hidden.shape}")
    print(f"Output shape: {score.shape}")
    print(f"Output score: {score.item():.4f}")
    print("Evaluator working correctly." if 0 <= score.item() <= 1 else "ERROR: score out of range")


if __name__ == "__main__":
    test_evaluator()
