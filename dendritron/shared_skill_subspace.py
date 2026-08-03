"""Dendritron's Universal/Shared-LoRA weight-space subspace.

This follows the central Share construction: stack task LoRA factors, center
their rank vectors, use SVD to obtain shared principal factors, and represent
each task through compact coordinates over those factors.

This is the model's 16-32 principal-direction Universal Subspace. The
dictionary remains an independently addressed CPU/disk knowledge bank.
Experts connect word/concept/task junctions to useful principal directions,
while fast coefficients remain episode-local state.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class LoRAFactors:
    task_id: str
    beta: np.ndarray
    alpha: np.ndarray

    def validate(self) -> tuple[int, int, int]:
        beta = np.asarray(self.beta)
        alpha = np.asarray(self.alpha)
        if beta.ndim != 2 or alpha.ndim != 2:
            raise ValueError("LoRA beta and alpha must both be matrices")
        output_width, rank = beta.shape
        if alpha.shape[0] != rank:
            raise ValueError(
                f"LoRA rank mismatch: beta {beta.shape}, alpha {alpha.shape}"
            )
        input_width = alpha.shape[1]
        return output_width, input_width, rank

    def delta(self) -> np.ndarray:
        self.validate()
        return np.asarray(self.beta) @ np.asarray(self.alpha)


@dataclass(frozen=True)
class AdapterCoefficients:
    """Fast coefficients selecting shared principal adapter directions."""

    task_id: str
    beta_coefficients: np.ndarray
    alpha_coefficients: np.ndarray


# Compatibility name for bundles created before the expert/coefficient split.
SkillCoordinates = AdapterCoefficients


@dataclass(frozen=True)
class SharedSkillBasis:
    beta_mean: np.ndarray
    alpha_mean: np.ndarray
    beta_basis: np.ndarray
    alpha_basis: np.ndarray
    beta_singular_values: np.ndarray
    alpha_singular_values: np.ndarray
    explained_variance_beta: np.ndarray
    explained_variance_alpha: np.ndarray

    @property
    def skill_rank(self) -> int:
        return int(self.beta_basis.shape[1])

    def project(self, factors: LoRAFactors) -> AdapterCoefficients:
        output_width, input_width, _ = factors.validate()
        if output_width != self.beta_basis.shape[0]:
            raise ValueError("LoRA output width differs from the skill basis")
        if input_width != self.alpha_basis.shape[0]:
            raise ValueError("LoRA input width differs from the skill basis")
        centered_beta = np.asarray(factors.beta) - self.beta_mean[:, None]
        centered_alpha = np.asarray(factors.alpha).T - self.alpha_mean[:, None]
        return AdapterCoefficients(
            task_id=factors.task_id,
            beta_coefficients=self.beta_basis.T @ centered_beta,
            alpha_coefficients=self.alpha_basis.T @ centered_alpha,
        )

    def reconstruct_factors(
        self,
        coefficients: AdapterCoefficients,
    ) -> LoRAFactors:
        beta = (
            self.beta_basis @ coefficients.beta_coefficients
            + self.beta_mean[:, None]
        )
        alpha_transposed = (
            self.alpha_basis @ coefficients.alpha_coefficients
            + self.alpha_mean[:, None]
        )
        return LoRAFactors(
            task_id=coefficients.task_id,
            beta=beta,
            alpha=alpha_transposed.T,
        )


def _explained_variance(singular_values: np.ndarray) -> np.ndarray:
    energy = np.asarray(singular_values, dtype=np.float64) ** 2
    total = float(energy.sum())
    return energy / total if total > 0 else np.zeros_like(energy)


def fit_shared_skill_basis(
    adapters: Sequence[LoRAFactors],
    *,
    skill_rank: int = 16,
) -> tuple[SharedSkillBasis, list[AdapterCoefficients]]:
    if not adapters:
        raise ValueError("At least one successful task adapter is required")
    shapes = [adapter.validate() for adapter in adapters]
    output_width, input_width, _ = shapes[0]
    if any(shape[:2] != (output_width, input_width) for shape in shapes):
        raise ValueError("All adapters must target the same operator shape")

    beta_stack = np.concatenate(
        [np.asarray(adapter.beta, dtype=np.float64) for adapter in adapters],
        axis=1,
    )
    alpha_stack = np.concatenate(
        [np.asarray(adapter.alpha, dtype=np.float64).T for adapter in adapters],
        axis=1,
    )
    maximum_rank = min(
        beta_stack.shape[0],
        alpha_stack.shape[0],
        beta_stack.shape[1],
        alpha_stack.shape[1],
    )
    if not 1 <= skill_rank <= maximum_rank:
        raise ValueError(
            f"skill_rank must be between 1 and {maximum_rank} for these adapters"
        )

    beta_mean = beta_stack.mean(axis=1)
    alpha_mean = alpha_stack.mean(axis=1)
    centered_beta = beta_stack - beta_mean[:, None]
    centered_alpha = alpha_stack - alpha_mean[:, None]
    beta_u, beta_s, _ = np.linalg.svd(centered_beta, full_matrices=False)
    alpha_u, alpha_s, _ = np.linalg.svd(centered_alpha, full_matrices=False)
    basis = SharedSkillBasis(
        beta_mean=beta_mean.astype(np.float32),
        alpha_mean=alpha_mean.astype(np.float32),
        beta_basis=beta_u[:, :skill_rank].astype(np.float32),
        alpha_basis=alpha_u[:, :skill_rank].astype(np.float32),
        beta_singular_values=beta_s.astype(np.float32),
        alpha_singular_values=alpha_s.astype(np.float32),
        explained_variance_beta=_explained_variance(beta_s).astype(np.float32),
        explained_variance_alpha=_explained_variance(alpha_s).astype(np.float32),
    )
    return basis, [basis.project(adapter) for adapter in adapters]


def orthogonal_update_residual(
    basis: np.ndarray,
    update_vectors: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Split new update vectors into existing-skill and x-part components."""
    directions = np.asarray(basis, dtype=np.float64)
    updates = np.asarray(update_vectors, dtype=np.float64)
    if directions.ndim != 2 or updates.ndim != 2:
        raise ValueError("basis and update_vectors must be matrices")
    if directions.shape[0] != updates.shape[0]:
        raise ValueError("basis and update vectors must share their first dimension")
    coordinates = directions.T @ updates
    residual = updates - directions @ coordinates
    return coordinates.astype(np.float32), residual.astype(np.float32)
