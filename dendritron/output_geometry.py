"""Softmax-free tied vocabulary distance and rank-margin objective."""

from __future__ import annotations

try:
    import torch
    from torch import Tensor, nn
    import torch.nn.functional as F
except ImportError as error:  # pragma: no cover - exercised on training host
    raise ImportError(
        "dendritron.output_geometry requires PyTorch. Install torch>=2.7."
    ) from error

from .geometric_attention import RMSNorm


class GeometricVocabularyHead(nn.Module):
    def __init__(self, width: int, *, epsilon: float = 1e-6) -> None:
        super().__init__()
        self.width = int(width)
        self.epsilon = float(epsilon)
        self.state_norm = RMSNorm(width, epsilon)
        self.state_operator = nn.Linear(width, width, bias=False)
        nn.init.eye_(self.state_operator.weight)

    def forward(self, hidden: Tensor, embedding_weight: Tensor) -> Tensor:
        if embedding_weight.ndim != 2 or embedding_weight.shape[1] != self.width:
            raise ValueError("Vocabulary embeddings have the wrong width")
        state = self.state_operator(self.state_norm(hidden))
        state = F.normalize(state, dim=-1)
        vocabulary = F.normalize(embedding_weight, dim=-1)
        displacement = state.unsqueeze(-2) - vocabulary.view(
            *([1] * (state.ndim - 1)),
            vocabulary.shape[0],
            vocabulary.shape[1],
        )
        return -displacement.square().mean(dim=-1)


def rank_margin_loss(
    logits: Tensor,
    targets: Tensor,
    *,
    margin: float = 0.2,
    hard_negatives: int = 32,
    ignore_index: int = -100,
) -> Tensor:
    """Hinge rank loss over the strongest raw-score negative tokens."""

    if logits.shape[:-1] != targets.shape:
        raise ValueError("targets must match logits without the vocabulary axis")
    if hard_negatives < 1:
        raise ValueError("hard_negatives must be positive")
    vocabulary_size = logits.shape[-1]
    count = min(int(hard_negatives), vocabulary_size - 1)
    if count < 1:
        raise ValueError("Vocabulary must contain at least two tokens")

    valid = targets != ignore_index
    safe_targets = targets.masked_fill(~valid, 0)
    positive = logits.gather(-1, safe_targets.unsqueeze(-1))
    negative_scores = logits.clone()
    negative_scores.scatter_(-1, safe_targets.unsqueeze(-1), -torch.inf)
    negatives = negative_scores.topk(count, dim=-1).values
    losses = F.relu(float(margin) - positive + negatives)
    losses = losses * valid.unsqueeze(-1)
    denominator = valid.sum().clamp_min(1) * count
    return losses.sum() / denominator
