"""Pure-Python schema and gates for the Qwen3.5 Dense benchmark matrix."""
from __future__ import annotations

import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any


class MatrixValidationError(ValueError):
    """Raised when a benchmark manifest cannot support strict comparison."""


@dataclass(frozen=True)
class ModelSpec:
    model_key: str
    official_model_id: str
    parameters_billions: float
    minimum_fp16_devices: int
    checkpoint_glob: str | None = None
    layers: int | None = None
    hidden_size: int | None = None
    head_size: int | None = None


@dataclass(frozen=True)
class TierSpec:
    tier_id: str
    rwkv: ModelSpec
    qwen: ModelSpec


@dataclass(frozen=True)
class WorkloadSpec:
    batch_size: int
    prompt_length: int
    decode_length: int


@dataclass(frozen=True)
class DenseMatrix:
    schema_version: int
    matrix_id: str
    scope: str
    sources: dict[str, str]
    precisions: tuple[str, ...]
    workloads: tuple[WorkloadSpec, ...]
    tiers: tuple[TierSpec, ...]


@dataclass(frozen=True)
class NormalizedRow:
    engine: str
    model_key: str
    model_family: str
    tier_id: str
    device_name: str
    device_count: int
    dtype: str
    batch_size: int
    prompt_length: int
    decode_length: int
    prefill_tokens_per_second: float | None
    decode_tokens_per_second: float | None
    peak_memory_mib: float | None
    correctness_passed: bool | None
    source_path: str
    memory_scope: str | None = None
    run_status: str = "ok"
    status_reason: str | None = None

    def with_updates(self, **changes: Any) -> "NormalizedRow":
        return replace(self, **changes)


@dataclass(frozen=True)
class EvaluatedRow:
    tier_id: str
    batch_size: int
    prompt_length: int
    decode_length: int
    dtype: str
    status: str
    reasons: tuple[str, ...]
    prefill_ratio: float | None = None
    decode_ratio: float | None = None
    memory_ratio: float | None = None


@dataclass(frozen=True)
class MatrixReport:
    matrix_id: str
    global_status: str
    rows: tuple[EvaluatedRow, ...]


def _positive_number(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
        raise MatrixValidationError(f"{field} must be positive")
    return float(value)


def _positive_int(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise MatrixValidationError(f"{field} must be a positive integer")
    return value


def _required_text(raw: dict[str, Any], key: str, field: str) -> str:
    value = raw.get(key)
    if not isinstance(value, str) or not value.strip():
        raise MatrixValidationError(f"{field} must be non-empty")
    return value


def _optional_positive_int(
    raw: dict[str, Any], key: str, field: str
) -> int | None:
    value = raw.get(key)
    return None if value is None else _positive_int(value, field)


def _parse_model(raw: Any, field: str) -> ModelSpec:
    if not isinstance(raw, dict):
        raise MatrixValidationError(f"{field} must be an object")
    return ModelSpec(
        model_key=_required_text(raw, "model_key", f"{field}.model_key"),
        official_model_id=_required_text(
            raw, "official_model_id", f"{field}.official_model_id"
        ),
        parameters_billions=_positive_number(
            raw.get("parameters_billions"), f"{field}.parameters_billions"
        ),
        minimum_fp16_devices=_positive_int(
            raw.get("minimum_fp16_devices"),
            f"{field}.minimum_fp16_devices",
        ),
        checkpoint_glob=raw.get("checkpoint_glob"),
        layers=_optional_positive_int(raw, "layers", f"{field}.layers"),
        hidden_size=_optional_positive_int(
            raw, "hidden_size", f"{field}.hidden_size"
        ),
        head_size=_optional_positive_int(
            raw, "head_size", f"{field}.head_size"
        ),
    )


def load_manifest(path: str | Path) -> DenseMatrix:
    """Load and strictly validate a five-tier comparison manifest."""
    manifest_path = Path(path)
    try:
        raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise MatrixValidationError(
            f"cannot read manifest {manifest_path}: {error}"
        ) from error
    if not isinstance(raw, dict):
        raise MatrixValidationError("manifest root must be an object")

    raw_workloads = raw.get("workloads")
    if not isinstance(raw_workloads, list) or not raw_workloads:
        raise MatrixValidationError("workloads must be a non-empty list")
    workloads = []
    for index, item in enumerate(raw_workloads):
        field = f"workloads[{index}]"
        if not isinstance(item, dict):
            raise MatrixValidationError(f"{field} must be an object")
        workloads.append(
            WorkloadSpec(
                batch_size=_positive_int(
                    item.get("batch_size"), f"{field}.batch_size"
                ),
                prompt_length=_positive_int(
                    item.get("prompt_length"), f"{field}.prompt_length"
                ),
                decode_length=_positive_int(
                    item.get("decode_length"), f"{field}.decode_length"
                ),
            )
        )
    if not {1, 4}.issubset({workload.batch_size for workload in workloads}):
        raise MatrixValidationError("workloads must include B1 and B4")

    raw_tiers = raw.get("tiers")
    if not isinstance(raw_tiers, list) or not raw_tiers:
        raise MatrixValidationError("tiers must be a non-empty list")
    tiers = []
    tier_ids: set[str] = set()
    for index, item in enumerate(raw_tiers):
        if not isinstance(item, dict):
            raise MatrixValidationError(f"tiers[{index}] must be an object")
        tier_id = _required_text(item, "tier_id", f"tiers[{index}].tier_id")
        if tier_id in tier_ids:
            raise MatrixValidationError(f"duplicate tier_id {tier_id}")
        tier_ids.add(tier_id)
        tiers.append(
            TierSpec(
                tier_id=tier_id,
                rwkv=_parse_model(item.get("rwkv"), f"{tier_id}.rwkv"),
                qwen=_parse_model(item.get("qwen"), f"{tier_id}.qwen"),
            )
        )

    sources = raw.get("sources")
    if not isinstance(sources, dict) or not all(
        isinstance(key, str) and isinstance(value, str) and value
        for key, value in sources.items()
    ):
        raise MatrixValidationError("sources must map names to URLs")
    precisions = raw.get("precisions")
    if not isinstance(precisions, list) or not precisions or not all(
        isinstance(value, str) and value for value in precisions
    ):
        raise MatrixValidationError("precisions must be a non-empty string list")

    return DenseMatrix(
        schema_version=_positive_int(raw.get("schema_version"), "schema_version"),
        matrix_id=_required_text(raw, "matrix_id", "matrix_id"),
        scope=_required_text(raw, "scope", "scope"),
        sources=dict(sources),
        precisions=tuple(precisions),
        workloads=tuple(workloads),
        tiers=tuple(tiers),
    )


def _find_row(
    rows: list[NormalizedRow],
    tier_id: str,
    family: str,
    workload: WorkloadSpec,
    dtype: str,
) -> NormalizedRow | None:
    matches = [
        row
        for row in rows
        if row.tier_id == tier_id
        and row.model_family == family
        and row.batch_size == workload.batch_size
        and row.prompt_length == workload.prompt_length
        and row.decode_length == workload.decode_length
        and row.dtype == dtype
    ]
    if len(matches) > 1:
        sources = ", ".join(row.source_path for row in matches)
        raise MatrixValidationError(
            f"duplicate {family} results for {tier_id} "
            f"B{workload.batch_size}P{workload.prompt_length}: {sources}"
        )
    return matches[0] if matches else None


def _evaluate_pair(
    tier: TierSpec,
    workload: WorkloadSpec,
    dtype: str,
    rwkv: NormalizedRow | None,
    qwen: NormalizedRow | None,
) -> EvaluatedRow:
    fields = {
        "tier_id": tier.tier_id,
        "batch_size": workload.batch_size,
        "prompt_length": workload.prompt_length,
        "decode_length": workload.decode_length,
        "dtype": dtype,
    }
    if rwkv is None or qwen is None:
        return EvaluatedRow(
            status="missing",
            reasons=("paired result is missing",),
            **fields,
        )
    blocked = [row for row in (rwkv, qwen) if row.run_status == "blocked"]
    if blocked:
        return EvaluatedRow(
            status="blocked",
            reasons=tuple(
                row.status_reason or f"{row.model_family} row is blocked"
                for row in blocked
            ),
            **fields,
        )
    failed = [row for row in (rwkv, qwen) if row.run_status != "ok"]
    if failed:
        return EvaluatedRow(
            status="fail",
            reasons=tuple(
                row.status_reason or f"{row.model_family} benchmark failed"
                for row in failed
            ),
            **fields,
        )

    reasons = []
    if rwkv.model_key != tier.rwkv.model_key or qwen.model_key != tier.qwen.model_key:
        reasons.append("paired model key differs from manifest")
    if (
        rwkv.device_name != qwen.device_name
        or rwkv.device_count != qwen.device_count
        or rwkv.dtype != qwen.dtype
    ):
        reasons.append("paired hardware or dtype differs")
    if (
        not rwkv.memory_scope
        or not qwen.memory_scope
        or rwkv.memory_scope != qwen.memory_scope
    ):
        reasons.append("paired memory scope is missing or differs")
    if rwkv.correctness_passed is not True:
        reasons.append("RWKV correctness failed")

    required_metrics = (
        rwkv.prefill_tokens_per_second,
        qwen.prefill_tokens_per_second,
        rwkv.decode_tokens_per_second,
        qwen.decode_tokens_per_second,
        rwkv.peak_memory_mib,
        qwen.peak_memory_mib,
    )
    if any(value is None for value in required_metrics):
        return EvaluatedRow(
            status="missing",
            reasons=tuple(reasons + ["required performance metric is missing"]),
            **fields,
        )

    assert rwkv.prefill_tokens_per_second is not None
    assert qwen.prefill_tokens_per_second is not None
    assert rwkv.decode_tokens_per_second is not None
    assert qwen.decode_tokens_per_second is not None
    assert rwkv.peak_memory_mib is not None
    assert qwen.peak_memory_mib is not None
    prefill_ratio = (
        rwkv.prefill_tokens_per_second / qwen.prefill_tokens_per_second
    )
    decode_ratio = rwkv.decode_tokens_per_second / qwen.decode_tokens_per_second
    memory_ratio = rwkv.peak_memory_mib / qwen.peak_memory_mib
    if prefill_ratio <= 1.0:
        reasons.append("RWKV prefill is not faster")
    if decode_ratio <= 1.0:
        reasons.append("RWKV decode is not faster")
    if memory_ratio > 1.0:
        reasons.append("RWKV peak memory is higher")
    return EvaluatedRow(
        status="fail" if reasons else "pass",
        reasons=tuple(reasons),
        prefill_ratio=prefill_ratio,
        decode_ratio=decode_ratio,
        memory_ratio=memory_ratio,
        **fields,
    )


def evaluate_matrix(
    matrix: DenseMatrix, rows: list[NormalizedRow]
) -> MatrixReport:
    """Evaluate every manifest row; missing/blocked work never passes."""
    evaluated = []
    for tier in matrix.tiers:
        for dtype in matrix.precisions:
            for workload in matrix.workloads:
                evaluated.append(
                    _evaluate_pair(
                        tier,
                        workload,
                        dtype,
                        _find_row(rows, tier.tier_id, "rwkv", workload, dtype),
                        _find_row(rows, tier.tier_id, "qwen", workload, dtype),
                    )
                )
    statuses = {row.status for row in evaluated}
    global_status = next(
        status
        for status in ("fail", "blocked", "missing", "pass")
        if status in statuses
    )
    return MatrixReport(
        matrix_id=matrix.matrix_id,
        global_status=global_status,
        rows=tuple(evaluated),
    )


def _infer_tier(
    matrix: DenseMatrix, model: str, family: str
) -> tuple[TierSpec, ModelSpec]:
    lowered = model.lower()
    matches = []
    for tier in matrix.tiers:
        spec = tier.rwkv if family == "rwkv" else tier.qwen
        scale = spec.model_key.rsplit("-", 1)[-1]
        if spec.model_key in lowered or scale in lowered:
            matches.append((tier, spec))
    if len(matches) != 1:
        raise MatrixValidationError(
            f"cannot uniquely map {family} model {model!r} to a matrix tier"
        )
    return matches[0]


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise MatrixValidationError(f"expected numeric metric, got {value!r}")
    return float(value)


def _common_result_fields(document: dict[str, Any]) -> dict[str, Any]:
    return {
        "device_name": str(
            document.get("device_name") or document.get("device") or "unknown"
        ),
        "device_count": int(document.get("device_count", 1)),
        "dtype": str(document.get("dtype", "fp16")),
        "run_status": str(document.get("run_status", "ok")),
        "status_reason": document.get("status_reason"),
    }


def normalize_result_document(
    matrix: DenseMatrix,
    document: dict[str, Any],
    source_path: str,
) -> list[NormalizedRow]:
    """Normalize one existing benchmark JSON document without inventing data."""
    benchmark = document.get("benchmark")
    model = document.get("model")
    if not isinstance(benchmark, str) or not isinstance(model, str):
        raise MatrixValidationError(
            f"{source_path}: benchmark and model must be strings"
        )
    common = _common_result_fields(document)
    if benchmark == "rwkv7_pth_prefill_npu":
        tier, spec = _infer_tier(matrix, model, "rwkv")
        shape = document.get("shape")
        correctness = document.get("correctness")
        if not isinstance(shape, dict) or not isinstance(correctness, dict):
            raise MatrixValidationError(
                f"{source_path}: RWKV result needs shape and correctness objects"
            )
        cosine = _optional_float(correctness.get("logits_cosine"))
        correctness_passed = bool(correctness.get("greedy_match")) and (
            cosine is not None and cosine >= 0.9999
        )
        return [
            NormalizedRow(
                engine="rwkv7_ascendc",
                model_key=spec.model_key,
                model_family="rwkv",
                tier_id=tier.tier_id,
                batch_size=int(shape.get("batch_size", 0)),
                prompt_length=int(shape.get("prompt_length", 0)),
                decode_length=int(document.get("decode_length", 0)),
                prefill_tokens_per_second=_optional_float(
                    document.get("layer_major_tokens_per_second")
                ),
                decode_tokens_per_second=_optional_float(
                    document.get("decode_tokens_per_second")
                ),
                peak_memory_mib=_optional_float(document.get("peak_memory_mib")),
                correctness_passed=correctness_passed,
                source_path=source_path,
                memory_scope=document.get("peak_memory_scope"),
                **common,
            )
        ]
    if benchmark == "qwen35_vllm_ascend":
        tier, spec = _infer_tier(matrix, model, "qwen")
        return [
            NormalizedRow(
                engine="vllm_ascend",
                model_key=spec.model_key,
                model_family="qwen",
                tier_id=tier.tier_id,
                batch_size=int(document.get("batch_size", 0)),
                prompt_length=int(document.get("prompt_length", 0)),
                decode_length=int(document.get("decode_length", 0)),
                prefill_tokens_per_second=_optional_float(
                    document.get("prefill_tokens_per_second")
                ),
                decode_tokens_per_second=_optional_float(
                    document.get("decode_tokens_per_second")
                ),
                peak_memory_mib=_optional_float(document.get("peak_memory_mib")),
                correctness_passed=None,
                source_path=source_path,
                memory_scope=document.get("peak_memory_scope"),
                **common,
            )
        ]
    if benchmark == "qwen35_transformers_npu":
        tier, spec = _infer_tier(matrix, model, "qwen")
        raw_rows = document.get("rows")
        if not isinstance(raw_rows, list):
            raise MatrixValidationError(
                f"{source_path}: Transformers result needs a rows list"
            )
        rows = []
        for raw in raw_rows:
            if not isinstance(raw, dict):
                raise MatrixValidationError(
                    f"{source_path}: Transformers rows must be objects"
                )
            rows.append(
                NormalizedRow(
                    engine="transformers_npu",
                    model_key=spec.model_key,
                    model_family="qwen",
                    tier_id=tier.tier_id,
                    batch_size=int(raw.get("batch_size", 0)),
                    prompt_length=int(raw.get("prompt_length", 0)),
                    decode_length=int(
                        raw.get(
                            "decode_length",
                            document.get("decode_length", 0),
                        )
                    ),
                    prefill_tokens_per_second=_optional_float(
                        raw.get("prefill_tokens_per_second")
                    ),
                    decode_tokens_per_second=_optional_float(
                        raw.get("decode_tokens_per_second")
                    ),
                    peak_memory_mib=_optional_float(raw.get("peak_memory_mib")),
                    correctness_passed=None,
                    source_path=source_path,
                    memory_scope=raw.get(
                        "peak_memory_scope",
                        document.get("peak_memory_scope"),
                    ),
                    **common,
                )
            )
        return rows
    if benchmark == "qwen35_dense_evidence":
        raw_rows = document.get("rows")
        if not isinstance(raw_rows, list):
            raise MatrixValidationError(
                f"{source_path}: dense evidence needs a rows list"
            )
        tiers = {tier.tier_id: tier for tier in matrix.tiers}
        rows = []
        for raw in raw_rows:
            if not isinstance(raw, dict):
                raise MatrixValidationError(
                    f"{source_path}: dense evidence rows must be objects"
                )
            tier_id = str(raw.get("tier_id", ""))
            family = str(raw.get("model_family", ""))
            if tier_id not in tiers or family not in ("rwkv", "qwen"):
                raise MatrixValidationError(
                    f"{source_path}: invalid evidence tier/family"
                )
            tier = tiers[tier_id]
            spec = tier.rwkv if family == "rwkv" else tier.qwen
            if raw.get("model_key") != spec.model_key:
                raise MatrixValidationError(
                    f"{source_path}: evidence model_key differs from manifest"
                )
            rows.append(
                NormalizedRow(
                    engine=str(raw.get("engine", "unknown")),
                    model_key=spec.model_key,
                    model_family=family,
                    tier_id=tier_id,
                    batch_size=int(raw.get("batch_size", 0)),
                    prompt_length=int(raw.get("prompt_length", 0)),
                    decode_length=int(raw.get("decode_length", 0)),
                    prefill_tokens_per_second=_optional_float(
                        raw.get("prefill_tokens_per_second")
                    ),
                    decode_tokens_per_second=_optional_float(
                        raw.get("decode_tokens_per_second")
                    ),
                    peak_memory_mib=_optional_float(raw.get("peak_memory_mib")),
                    correctness_passed=raw.get("correctness_passed"),
                    source_path=(
                        f"{source_path}#{raw.get('remote_source', tier_id)}"
                    ),
                    memory_scope=raw.get(
                        "peak_memory_scope",
                        document.get("peak_memory_scope"),
                    ),
                    run_status=str(raw.get("run_status", "ok")),
                    status_reason=raw.get("status_reason"),
                    device_name=str(
                        raw.get("device_name", common["device_name"])
                    ),
                    device_count=int(
                        raw.get("device_count", common["device_count"])
                    ),
                    dtype=str(raw.get("dtype", common["dtype"])),
                )
            )
        return rows
    raise MatrixValidationError(
        f"{source_path}: unsupported benchmark {benchmark!r}"
    )


def _ratio(value: float | None) -> str:
    return "-" if value is None else f"{value:.3f}x"


def render_markdown(report: MatrixReport) -> str:
    """Render a compact human review table from a strict matrix report."""
    lines = [
        f"# {report.matrix_id}",
        "",
        f"Global status: **{report.global_status.upper()}**",
        "",
        "| Tier | Workload | Status | Prefill | Decode | Memory | Reasons |",
        "| --- | --- | --- | ---: | ---: | ---: | --- |",
    ]
    for row in report.rows:
        workload = (
            f"B{row.batch_size}/P{row.prompt_length}/D{row.decode_length}/"
            f"{row.dtype}"
        )
        lines.append(
            "| %s | %s | %s | %s | %s | %s | %s |"
            % (
                row.tier_id,
                workload,
                row.status,
                _ratio(row.prefill_ratio),
                _ratio(row.decode_ratio),
                _ratio(row.memory_ratio),
                "; ".join(row.reasons) or "-",
            )
        )
    return "\n".join(lines) + "\n"
