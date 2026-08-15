"""Validation contract for the reviewed punctuation-corrected Engram bank."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping


CORRECTED_STAGE2_ROOT = Path("/data/dendritron-stage2-punctuation-v2")
CORRECTED_JTD_ROOT = Path("/data/dendritron-stage4-jtd-punctuation-v2")

EXPECTED_ROWS = {
    "bigrams": 500_000,
    "trigrams": 500_000,
}
EXPECTED_MIGRATION = {
    "reusable_rows_copied": 809_775,
    "new_rows_extracted": 190_225,
    "retired_rows": 190_225,
    "total_corrected_rows": 1_000_000,
}
EXPECTED_HIDDEN_STATE_LAYERS = [8, 24]
EXPECTED_TENSOR_KEYS = ["layer08", "layer24"]


def validate_corrected_stage2_manifest(
    manifest: Mapping[str, Any],
    *,
    manifest_path: Path | None = None,
) -> dict[str, Any]:
    """Require the exact reviewed punctuation-v2 Stage-2 data contract.

    This gate prevents the surface index and latent-anchor exporter from
    silently falling back to the original punctuation-crossing inventory.
    The returned record is suitable for embedding in downstream manifests.
    """

    location = f" at {manifest_path}" if manifest_path is not None else ""
    if manifest.get("completed") is not True:
        raise ValueError(f"Corrected Stage-2 manifest is incomplete{location}")

    addressing = manifest.get("addressing", {})
    if addressing.get("punctuation_boundary_aware") is not True:
        raise ValueError(
            "Stage-2 input lacks the punctuation-boundary correction"
            f"{location}"
        )

    states = manifest.get("states", {})
    observed_layers = [int(value) for value in states.get("hidden_state_layers", [])]
    if observed_layers != EXPECTED_HIDDEN_STATE_LAYERS:
        raise ValueError(
            "Corrected Stage-2 hidden-state layers differ: "
            f"expected={EXPECTED_HIDDEN_STATE_LAYERS}, observed={observed_layers}"
        )

    sharding = manifest.get("sharding", {})
    observed_keys = [str(value) for value in sharding.get("tensor_keys", [])]
    if observed_keys != EXPECTED_TENSOR_KEYS:
        raise ValueError(
            "Corrected Stage-2 tensor keys differ: "
            f"expected={EXPECTED_TENSOR_KEYS}, observed={observed_keys}"
        )
    if sharding.get("both_layer_views_share_each_engram_row") is not True:
        raise ValueError("Stage-2 layer-8/layer-24 row pairing is unconfirmed")

    tables = manifest.get("tables", {})
    observed_rows: dict[str, int] = {}
    for name, expected in EXPECTED_ROWS.items():
        table = tables.get(name, {})
        observed = int(table.get("rows", -1))
        if observed != expected:
            raise ValueError(
                f"Corrected Stage-2 {name} rows differ: "
                f"expected={expected}, observed={observed}"
            )
        table_keys = [str(value) for value in table.get("tensor_keys", [])]
        if table_keys != EXPECTED_TENSOR_KEYS:
            raise ValueError(
                f"Corrected Stage-2 {name} tensor keys differ: "
                f"expected={EXPECTED_TENSOR_KEYS}, observed={table_keys}"
            )
        observed_rows[name] = observed

    migration = manifest.get("migration", {})
    observed_migration = {
        key: int(migration.get(key, -1)) for key in EXPECTED_MIGRATION
    }
    if observed_migration != EXPECTED_MIGRATION:
        raise ValueError(
            "Corrected Stage-2 migration totals differ: "
            f"expected={EXPECTED_MIGRATION}, observed={observed_migration}"
        )

    return {
        "contract": "reviewed_punctuation_v2_stage2",
        "punctuation_boundary_aware": True,
        "rows": observed_rows,
        "hidden_state_layers": observed_layers,
        "tensor_keys": observed_keys,
        "migration": observed_migration,
    }
