"""Optional dictionary-vector PCA utilities.

Each dictionary sense retains its full layer-2 vector and ordered
definition-word links in the sparse CPU/disk bank. These helpers can measure
the geometry of those vectors for diagnostics.

Dendritron's Universal/Shared-LoRA subspace is the 16-32D weight-space basis
built from successful task adapters. It remains the runtime skill subspace.

This module uses streaming sufficient statistics. It can therefore fit a full
2,048D covariance matrix without materializing every dictionary row at once.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Iterator, Sequence

import numpy as np


DEFAULT_KNOWLEDGE_RANK_CANDIDATES = (16, 32, 64, 128, 256, 512, 1024, 2048)
DEFAULT_EXPLAINED_VARIANCE_TARGET = 0.98


@dataclass(frozen=True)
class SourceMoments:
    name: str
    count: int
    mean: np.ndarray
    second_moment: np.ndarray


@dataclass(frozen=True)
class UniversalKnowledgeBasis:
    mean: np.ndarray
    components: np.ndarray
    eigenvalues: np.ndarray
    explained_variance_ratio: np.ndarray
    source_names: tuple[str, ...]
    source_weights: np.ndarray
    l2_normalized_input: bool

    @property
    def rank(self) -> int:
        return int(self.components.shape[1])

    @property
    def width(self) -> int:
        return int(self.components.shape[0])

    def project(self, vectors: np.ndarray) -> np.ndarray:
        values = np.asarray(vectors, dtype=np.float32)
        if self.l2_normalized_input:
            values = l2_normalize_rows(values)
        return (values - self.mean) @ self.components

    def reconstruct(self, coordinates: np.ndarray) -> np.ndarray:
        return np.asarray(coordinates) @ self.components.T + self.mean


@dataclass(frozen=True)
class KnowledgeRankSelection:
    """Measured rank decision for an optional definition-vector diagnostic."""

    rank: int
    explained_variance: float
    target: float
    target_reached: bool
    candidates: tuple[int, ...]


def select_knowledge_rank(
    explained_variance_ratio: np.ndarray,
    *,
    target: float = DEFAULT_EXPLAINED_VARIANCE_TARGET,
    candidates: Sequence[int] = DEFAULT_KNOWLEDGE_RANK_CANDIDATES,
    minimum_rank: int = 16,
    maximum_rank: int | None = None,
) -> KnowledgeRankSelection:
    """Choose the smallest measured candidate rank reaching ``target``.

    The diagnostic tests rank 16 first, then expands to the first measured
    candidate that reaches the requested variance target.
    """
    ratio = np.asarray(explained_variance_ratio, dtype=np.float64)
    if ratio.ndim != 1 or ratio.size == 0:
        raise ValueError("explained_variance_ratio must be a nonempty vector")
    if np.any(ratio < 0) or not np.isfinite(ratio).all():
        raise ValueError("explained_variance_ratio must be finite and nonnegative")
    if not 0.0 < target <= 1.0:
        raise ValueError("target must be in (0, 1]")
    upper = ratio.size if maximum_rank is None else min(int(maximum_rank), ratio.size)
    usable = tuple(
        sorted(
            {
                int(rank)
                for rank in candidates
                if minimum_rank <= int(rank) <= upper
            }
        )
    )
    if not usable:
        raise ValueError("No candidate ranks fall inside the requested range")

    cumulative = np.cumsum(ratio)
    selected = usable[-1]
    reached = False
    for rank in usable:
        if float(cumulative[rank - 1]) >= target:
            selected = rank
            reached = True
            break
    return KnowledgeRankSelection(
        rank=selected,
        explained_variance=float(cumulative[selected - 1]),
        target=float(target),
        target_reached=reached,
        candidates=usable,
    )


def l2_normalize_rows(values: np.ndarray, epsilon: float = 1e-12) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    return values / np.maximum(norms, epsilon)


def iter_array_batches(
    arrays: Iterable[np.ndarray],
    *,
    width: int,
) -> Iterator[np.ndarray]:
    for array in arrays:
        values = np.asarray(array)
        if values.ndim != 2 or values.shape[1] != width:
            raise ValueError(
                f"Expected vector batches shaped [N, {width}], found {values.shape}"
            )
        if values.shape[0]:
            yield values


def accumulate_source_moments(
    name: str,
    batches: Iterable[np.ndarray],
    *,
    width: int,
    l2_normalize: bool = True,
) -> SourceMoments:
    count = 0
    vector_sum = np.zeros(width, dtype=np.float64)
    outer_sum = np.zeros((width, width), dtype=np.float64)

    for batch in iter_array_batches(batches, width=width):
        values = np.asarray(batch, dtype=np.float64)
        if l2_normalize:
            values = l2_normalize_rows(values)
        count += values.shape[0]
        vector_sum += values.sum(axis=0)
        outer_sum += values.T @ values

    if count == 0:
        raise ValueError(f"Knowledge source {name!r} has zero rows")
    return SourceMoments(
        name=name,
        count=count,
        mean=vector_sum / count,
        second_moment=outer_sum / count,
    )


def fit_universal_knowledge_basis(
    moments: Sequence[SourceMoments],
    *,
    rank: int,
    source_weights: Sequence[float] | None = None,
    l2_normalized_input: bool = True,
) -> UniversalKnowledgeBasis:
    """Fit a balanced PCA basis from per-source sufficient statistics."""
    if not moments:
        raise ValueError("At least one knowledge source is required")
    width = int(moments[0].mean.shape[0])
    if any(item.mean.shape != (width,) for item in moments):
        raise ValueError("All knowledge sources must use the same vector width")
    if not 1 <= rank <= width:
        raise ValueError(f"rank must be between 1 and {width}")

    if source_weights is None:
        weights = np.full(len(moments), 1.0 / len(moments), dtype=np.float64)
    else:
        weights = np.asarray(source_weights, dtype=np.float64)
        if weights.shape != (len(moments),):
            raise ValueError("source_weights must have one value per source")
        if np.any(weights < 0) or weights.sum() <= 0:
            raise ValueError("source_weights must be nonnegative with positive mass")
        weights = weights / weights.sum()

    global_mean = sum(
        weight * item.mean for weight, item in zip(weights, moments, strict=True)
    )
    global_second = sum(
        weight * item.second_moment
        for weight, item in zip(weights, moments, strict=True)
    )
    covariance = global_second - np.outer(global_mean, global_mean)
    covariance = (covariance + covariance.T) * 0.5

    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = np.maximum(eigenvalues[order], 0.0)
    eigenvectors = eigenvectors[:, order]
    total = float(eigenvalues.sum())
    ratio = eigenvalues / total if total > 0 else np.zeros_like(eigenvalues)

    return UniversalKnowledgeBasis(
        mean=global_mean.astype(np.float32),
        components=eigenvectors[:, :rank].astype(np.float32),
        eigenvalues=eigenvalues[:rank].astype(np.float32),
        explained_variance_ratio=ratio[:rank].astype(np.float32),
        source_names=tuple(item.name for item in moments),
        source_weights=weights.astype(np.float32),
        l2_normalized_input=l2_normalized_input,
    )


def reconstruction_cosine(
    basis: UniversalKnowledgeBasis,
    vectors: np.ndarray,
    epsilon: float = 1e-12,
) -> np.ndarray:
    values = np.asarray(vectors, dtype=np.float32)
    target = l2_normalize_rows(values) if basis.l2_normalized_input else values
    restored = basis.reconstruct(basis.project(values))
    numerator = np.sum(target * restored, axis=1)
    denominator = np.linalg.norm(target, axis=1) * np.linalg.norm(restored, axis=1)
    return numerator / np.maximum(denominator, epsilon)
