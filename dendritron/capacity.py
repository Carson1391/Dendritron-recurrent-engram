"""Fixed 25/75 conditional sparse-capacity accounting.

The ledger covers appendable or conditionally addressed memory and compute
capacity.  The two recurrent physical blocks are shared active substrate and
are reported separately.  Counting them inside the sparse allocation would
make the ratio depend on the unroll count even though the stored blocks remain
unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass


FIXED_MEMORY_FRACTION = 0.25
FIXED_COMPUTE_FRACTION = 0.75


@dataclass(frozen=True)
class SparseCapacityLedger:
    memory_parameters: int
    compute_parameters: int
    shared_core_parameters: int = 0

    def __post_init__(self) -> None:
        for name, value in (
            ("memory_parameters", self.memory_parameters),
            ("compute_parameters", self.compute_parameters),
            ("shared_core_parameters", self.shared_core_parameters),
        ):
            if int(value) < 0:
                raise ValueError(f"{name} must be nonnegative")
        if self.memory_parameters + self.compute_parameters == 0:
            raise ValueError("sparse capacity cannot be empty")

    @property
    def sparse_total(self) -> int:
        return self.memory_parameters + self.compute_parameters

    @property
    def memory_fraction(self) -> float:
        return self.memory_parameters / self.sparse_total

    @property
    def compute_fraction(self) -> float:
        return self.compute_parameters / self.sparse_total

    @property
    def required_compute_parameters(self) -> int:
        return 3 * self.memory_parameters

    @property
    def compute_parameter_gap(self) -> int:
        return self.required_compute_parameters - self.compute_parameters

    @property
    def exact_25_75(self) -> bool:
        return self.compute_parameters == self.required_compute_parameters

    def assert_fixed_split(self) -> None:
        if not self.exact_25_75:
            raise ValueError(
                "Conditional sparse capacity violates the fixed 25/75 split: "
                f"memory={self.memory_parameters:,}, "
                f"compute={self.compute_parameters:,}, "
                f"required_compute={self.required_compute_parameters:,}"
            )

    def as_dict(self) -> dict[str, int | float | bool]:
        return {
            "memory_parameters": self.memory_parameters,
            "compute_parameters": self.compute_parameters,
            "shared_core_parameters": self.shared_core_parameters,
            "sparse_total": self.sparse_total,
            "memory_fraction": self.memory_fraction,
            "compute_fraction": self.compute_fraction,
            "required_compute_parameters": self.required_compute_parameters,
            "compute_parameter_gap": self.compute_parameter_gap,
            "exact_25_75": self.exact_25_75,
        }


def required_compute_for_memory(memory_parameters: int) -> int:
    if memory_parameters < 0:
        raise ValueError("memory_parameters must be nonnegative")
    return 3 * int(memory_parameters)


@dataclass(frozen=True)
class ExpertCapacityPlan:
    memory_parameters: int
    fixed_compute_parameters: int
    expert_count: int
    expert_hidden_width: int
    expert_parameters: int
    total_compute_parameters: int
    exact_25_75: bool
    absolute_gap: int


def solve_expert_geometry(
    *,
    memory_parameters: int,
    fixed_compute_parameters: int,
    model_width: int,
    branch_count: int,
    block_count: int = 2,
    minimum_experts: int = 1,
    maximum_experts: int = 4096,
    minimum_hidden_width: int = 32,
    maximum_hidden_width: int = 8192,
) -> ExpertCapacityPlan:
    """Choose expert count and hidden width for the fixed global split.

    Each expert branch owns three primary matrices: content, gate, and output.
    The search first returns an exact integer geometry.  When table granularity
    prevents equality, it returns the closest admissible geometry and exposes
    the scalar gap for a deliberate table-size adjustment.
    """

    values = {
        "memory_parameters": memory_parameters,
        "fixed_compute_parameters": fixed_compute_parameters,
        "model_width": model_width,
        "branch_count": branch_count,
        "block_count": block_count,
        "minimum_experts": minimum_experts,
        "maximum_experts": maximum_experts,
        "minimum_hidden_width": minimum_hidden_width,
        "maximum_hidden_width": maximum_hidden_width,
    }
    if any(int(value) < 0 for value in values.values()):
        raise ValueError("capacity-planner values must be nonnegative")
    if min(model_width, branch_count, block_count, minimum_experts) < 1:
        raise ValueError("widths, branches, blocks, and expert counts must be positive")
    if maximum_experts < minimum_experts:
        raise ValueError("maximum_experts must reach minimum_experts")
    if maximum_hidden_width < minimum_hidden_width:
        raise ValueError("maximum_hidden_width must reach minimum_hidden_width")

    target = required_compute_for_memory(memory_parameters)
    remaining = target - int(fixed_compute_parameters)
    if remaining <= 0:
        return ExpertCapacityPlan(
            memory_parameters=int(memory_parameters),
            fixed_compute_parameters=int(fixed_compute_parameters),
            expert_count=0,
            expert_hidden_width=0,
            expert_parameters=0,
            total_compute_parameters=int(fixed_compute_parameters),
            exact_25_75=remaining == 0,
            absolute_gap=abs(remaining),
        )

    coefficient = 3 * int(block_count) * int(branch_count) * int(model_width)
    best: ExpertCapacityPlan | None = None
    for expert_count in range(int(minimum_experts), int(maximum_experts) + 1):
        divisor = coefficient * expert_count
        quotient, remainder = divmod(remaining, divisor)
        candidate_widths = {quotient, quotient + int(remainder > 0)}
        for hidden_width in candidate_widths:
            if not minimum_hidden_width <= hidden_width <= maximum_hidden_width:
                continue
            expert_parameters = divisor * hidden_width
            total_compute = int(fixed_compute_parameters) + expert_parameters
            gap = abs(target - total_compute)
            candidate = ExpertCapacityPlan(
                memory_parameters=int(memory_parameters),
                fixed_compute_parameters=int(fixed_compute_parameters),
                expert_count=expert_count,
                expert_hidden_width=hidden_width,
                expert_parameters=expert_parameters,
                total_compute_parameters=total_compute,
                exact_25_75=gap == 0,
                absolute_gap=gap,
            )
            if candidate.exact_25_75:
                return candidate
            if best is None or (
                candidate.absolute_gap,
                candidate.expert_count,
                candidate.expert_hidden_width,
            ) < (
                best.absolute_gap,
                best.expert_count,
                best.expert_hidden_width,
            ):
                best = candidate
    if best is None:
        raise ValueError("No expert geometry falls inside the requested bounds")
    return best
