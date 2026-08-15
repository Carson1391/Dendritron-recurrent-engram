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
