"""Typed reasoning branch operators for expert-owned deductive branches.

Each operator is shared execution code: one implementation reused by every
expert that carries a matching ``BranchSpec(operator=...)``.  The operator
binds semantic roles (premises, conclusion, contradictions) to current hidden
states and stored trainable anchors, constructs a HarMax contraction pool,
and returns the signed movement, harmonic residual, and branch evidence.

The dependency chain is:

    current state
    -> select LoRA skills
    -> each skill exposes its adjacent experts
    -> selected experts expose their own branches
    -> those branches bind memory and propose movements

Branches belong to experts.  Skills determine eligible experts through
learned adjacency; selected experts determine the available branch
specifications.  The deductive operator is invoked only from an active expert
carrying a deductive ``BranchSpec``.
"""

from __future__ import annotations

from dataclasses import dataclass

try:
    import torch
    from torch import Tensor, nn
    import torch.nn.functional as F
except ImportError as error:  # pragma: no cover - exercised on training host
    raise ImportError(
        "dendritron.branch_operators requires PyTorch. Install torch>=2.7."
    ) from error

from .geometric_attention import HarMaxContraction


@dataclass(frozen=True)
class BranchOutcome:
    """Return value from a typed branch instantiation.

    Attributes:
        movement: Delta = -nabla_h rho, the signed HarMax movement [B, T, D].
        residual: rho, the unresolved harmonic residual [B, T].
        evidence: q, signed branch evidence in [-1, 1] scaled by (1+rho)^-1 [B, T].
        confidence: 1 / (1 + rho), the branch confidence [B, T].
    """

    movement: Tensor
    residual: Tensor
    evidence: Tensor
    confidence: Tensor


def gather_causal_premises(
    hidden: Tensor,
    max_premises: int,
) -> tuple[Tensor, Tensor]:
    """Vectorized version of gather_causal_premises.

    Args:
        hidden: [B, T, D] current live hidden state.
        max_premises: maximum number of causal premise anchors.

    Returns:
        premises: [B, T, P, D] where P = max_premises.
        valid: [B, T, P] bool mask.
    """
    B, T, D = hidden.shape
    P = int(max_premises)

    positions = torch.arange(T, device=hidden.device)
    slots = torch.arange(P, device=hidden.device)
    # source_indices[t, s] = t - P + s (most recent P positions before t)
    source_indices = positions.unsqueeze(1) + slots.unsqueeze(0) - P  # [T, P]
    source_valid = (source_indices >= 0) & (source_indices < positions.unsqueeze(1))  # [T, P]

    safe_source = source_indices.clamp(min=0, max=T - 1)  # [T, P]

    # Expand for batch: [B, T, P, D]
    # hidden[b, src, d] for each (b, t, s)
    # Use advanced indexing: hidden[:, safe_source, :] -> [B, T, P, D]
    premises = hidden[:, safe_source, :]  # [B, T, P, D]
    valid = source_valid.unsqueeze(0).expand(B, -1, -1)  # [B, T, P]

    # Zero out invalid premises
    premises = premises * valid.unsqueeze(-1).to(hidden.dtype)

    return premises, valid


class DeductiveBranchOperator(nn.Module):
    """Shared deductive branch execution code.

    From the master specification (Section 10.6):

        Pool construction: Bind all required premises, the proposed conclusion,
        and contradiction anchors.

        Branch action: Contract jointly supported premises and conclusion;
        repel conclusions that violate any bound premise.

    This operator is shared across all experts that carry a deductive
    ``BranchSpec``.  One instance serves thousands of expert junctions.
    """

    def __init__(
        self,
        width: int,
        *,
        harmonic_exponent: float = 2.0,
        epsilon: float = 1e-6,
    ) -> None:
        super().__init__()
        self.width = int(width)
        self.epsilon = float(epsilon)
        # HarMax contraction for explicit pool evaluation.
        # Large candidate_window and top_k so all bound anchors participate.
        self.contraction = HarMaxContraction(
            width,
            max_sequence_length=4096,
            candidate_window=4096,
            top_k=4096,
            harmonic_exponent=harmonic_exponent,
            epsilon=epsilon,
        )

    def forward(
        self,
        hidden: Tensor,
        premise_anchors: Tensor,
        premise_valid: Tensor,
        conclusion_anchor: Tensor,
        contradiction_anchors: Tensor,
        premise_evidence: Tensor,
        conclusion_evidence: Tensor,
    ) -> BranchOutcome:
        """Execute the deductive branch.

        Args:
            hidden: [B, T, D] current live hidden state (query).
            premise_anchors: [B, T, P, D] bound premise vectors from earlier positions.
            premise_valid: [B, T, P] bool mask for valid premise slots.
            conclusion_anchor: [B, T, D] the proposed conclusion direction.
            contradiction_anchors: [B, T, X, D] contradiction anchor vectors.
            premise_evidence: [B, T, P] positive evidence weights for premises.
            conclusion_evidence: [B, T] positive evidence weight for conclusion.

        Returns:
            BranchOutcome with movement, residual, evidence, and confidence.
        """
        B, T, D = hidden.shape
        P = premise_anchors.shape[-2]
        X = contradiction_anchors.shape[-2]

        # Build the HarMax pool: [premises, conclusion, contradictions]
        # conclusion is a single anchor expanded to [B, T, 1, D]
        conclusion_expanded = conclusion_anchor.unsqueeze(-2)  # [B, T, 1, D]
        anchors = torch.cat(
            [premise_anchors, conclusion_expanded, contradiction_anchors],
            dim=-2,
        )  # [B, T, P+1+X, D]

        # Evidence: premises and conclusion are supported; contradictions are not.
        # Contradictions get zero evidence and supported=False so they only
        # contribute distance mass (repulsion) via the HarMax signed coefficients.
        conclusion_ev_expanded = conclusion_evidence.unsqueeze(-1)  # [B, T, 1]
        contradiction_ev = hidden.new_zeros(B, T, X)
        evidence = torch.cat(
            [premise_evidence, conclusion_ev_expanded, contradiction_ev],
            dim=-1,
        )  # [B, T, P+1+X]

        # Supported mask: premises and conclusion are supported, contradictions are not
        supported = torch.cat(
            [
                premise_valid,  # [B, T, P]
                torch.ones(B, T, 1, dtype=torch.bool, device=hidden.device),
                torch.zeros(B, T, X, dtype=torch.bool, device=hidden.device),
            ],
            dim=-1,
        )  # [B, T, P+1+X]

        # Valid mask: premises use premise_valid, conclusion and contradictions always valid
        valid = torch.cat(
            [
                premise_valid,
                torch.ones(B, T, 1 + X, dtype=torch.bool, device=hidden.device),
            ],
            dim=-1,
        )  # [B, T, P+1+X]

        # Evaluate the HarMax contraction pool
        movement, stats = self.contraction.contract_pool(
            hidden,
            anchors,
            evidence=evidence,
            supported=supported,
            valid=valid,
        )

        # Compute the relation direction: premise centroid -> conclusion
        # v_hat_R is the Euclidean unit direction induced by the branch's
        # bound source (premises) and conclusion anchors.
        # Only use valid premises for the centroid.
        premise_mask = premise_valid.to(hidden.dtype).unsqueeze(-1)  # [B, T, P, 1]
        premise_count = premise_mask.sum(dim=-2).clamp_min(1.0)  # [B, T, 1]
        premise_centroid = (premise_anchors * premise_mask).sum(dim=-2) / premise_count  # [B, T, D]
        relation_direction = conclusion_anchor - premise_centroid  # [B, T, D]
        v_hat = F.normalize(relation_direction, dim=-1)  # [B, T, D]

        # Signed evidence: zeta = <Delta, v_hat> / (epsilon + ||Delta||)
        # This gives positive evidence when the derivative moves along the
        # specified relation and opposing evidence when it moves against.
        delta_norm = movement.norm(dim=-1)  # [B, T]
        zeta = (movement * v_hat).sum(dim=-1) / (self.epsilon + delta_norm)  # [B, T]

        # Harmonic residual from the contraction
        rho = stats.harmonic_residual  # [B, T]

        # q = zeta / (1 + rho)
        # The branch evidence is damped by the unresolved residual.
        q = zeta / (1.0 + rho.abs())  # [B, T]

        # Confidence: 1 / (1 + rho)
        confidence = (1.0 + rho).reciprocal()  # [B, T]

        return BranchOutcome(
            movement=movement,
            residual=rho,
            evidence=q,
            confidence=confidence,
        )


class DeductiveBranchInstance(nn.Module):
    """Per-expert trainable parameters for one deductive branch.

    Stores the conclusion anchor and contradiction anchors as trainable
    vectors.  Premises bind to live hidden states at runtime through the
    shared ``DeductiveBranchOperator``.

    The branch specification (``BranchSpec``) is metadata that identifies
    this as a deductive branch; the trainable parameters live here.
    """

    def __init__(
        self,
        width: int,
        *,
        max_premises: int = 8,
        max_contradictions: int = 4,
        harmonic_exponent: float = 2.0,
        epsilon: float = 1e-6,
    ) -> None:
        super().__init__()
        self.width = int(width)
        self.max_premises = int(max_premises)
        self.max_contradictions = int(max_contradictions)
        self.epsilon = float(epsilon)

        # Trainable conclusion anchor: the proposed conclusion direction
        self.conclusion_anchor = nn.Parameter(torch.empty(width))
        nn.init.normal_(self.conclusion_anchor, std=width**-0.5)

        # Trainable contradiction anchors: directions to repel
        self.contradiction_anchors = nn.Parameter(
            torch.empty(max_contradictions, width)
        )
        nn.init.normal_(self.contradiction_anchors, std=width**-0.5)

        # Evidence weights for premises (one per premise slot)
        self.premise_evidence_raw = nn.Parameter(torch.zeros(max_premises))

        # Evidence weight for the conclusion
        self.conclusion_evidence_raw = nn.Parameter(torch.tensor(0.0))

        # Branch gate: controls the strength of the branch movement
        self.gate = nn.Parameter(torch.tensor(1e-3))

    def forward(
        self,
        hidden: Tensor,
        operator: DeductiveBranchOperator,
    ) -> tuple[Tensor, BranchOutcome]:
        """Execute this deductive branch instance.

        Args:
            hidden: [B, T, D] current live hidden state.
            operator: shared DeductiveBranchOperator.

        Returns:
            gated_movement: [B, T, D] gated branch movement for the contraction residual.
            outcome: full BranchOutcome for the expert soma.
        """
        B, T, D = hidden.shape
        P = self.max_premises
        X = self.max_contradictions

        # Gather premise anchors from earlier hidden states (causal)
        premise_anchors, premise_valid = gather_causal_premises(
            hidden, max_premises=P
        )  # [B, T, P, D], [B, T, P]

        # Broadcast conclusion anchor to [B, T, D]
        conclusion = self.conclusion_anchor.view(1, 1, D).expand(B, T, D)

        # Broadcast contradiction anchors to [B, T, X, D]
        contradictions = self.contradiction_anchors.view(1, 1, X, D).expand(B, T, X, D)

        # Apply softplus to evidence weights for positivity
        premise_evidence = F.softplus(self.premise_evidence_raw).view(1, 1, P).expand(B, T, P)
        conclusion_evidence = F.softplus(self.conclusion_evidence_raw).expand(B, T)

        # Execute the shared deductive operator
        outcome = operator(
            hidden=hidden,
            premise_anchors=premise_anchors,
            premise_valid=premise_valid,
            conclusion_anchor=conclusion,
            contradiction_anchors=contradictions,
            premise_evidence=premise_evidence,
            conclusion_evidence=conclusion_evidence,
        )

        # Gate the movement
        gated_movement = torch.tanh(self.gate) * outcome.movement

        return gated_movement, outcome


def combine_branch_movements(
    movements: list[Tensor],
    evidences: list[Tensor],
    epsilon: float = 1e-6,
) -> Tensor:
    """Combine branch movements with signed L1 normalization (expert soma).

    From the master specification (Section 10.6):

        Delta_e = sum(q_j * Delta_j) / (epsilon + sum(|q_j|))

    Args:
        movements: list of [B, T, D] tensors, one per active branch.
        evidences: list of [B, T] tensors, signed branch evidence.
        epsilon: numerical stability constant.

    Returns:
        [B, T, D] combined movement.
    """
    if not movements:
        # No active branches: return zero movement
        raise ValueError("combine_branch_movements requires at least one branch")

    stacked_movements = torch.stack(movements, dim=-1)  # [B, T, D, J]
    stacked_evidences = torch.stack(evidences, dim=-1)  # [B, T, J]

    # Signed L1 normalization: q_j * Delta_j / (epsilon + sum(|q_j|))
    abs_sum = stacked_evidences.abs().sum(dim=-1, keepdim=True)  # [B, T, 1]
    weights = stacked_evidences / (abs_sum + epsilon)  # [B, T, J]

    # Weighted sum: sum_j w_j * Delta_j
    combined = (stacked_movements * weights.unsqueeze(-2)).sum(dim=-1)  # [B, T, D]

    return combined
