"""Analyze RWKV-7 vs Qwen3.5 Dense result JSON with strict full gates."""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from model_matrix import (
    evaluate_matrix,
    load_manifest,
    normalize_result_document,
    render_markdown,
)


HERE = Path(__file__).resolve().parent
DEFAULT_MANIFEST = HERE / "qwen35_dense_matrix.json"


def _result_paths(values: list[str]) -> list[Path]:
    paths = []
    for value in values:
        path = Path(value)
        if path.is_dir():
            paths.extend(sorted(path.rglob("*.json")))
        else:
            paths.append(path)
    return paths


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("results", nargs="+", help="result JSON files or directories")
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--json-output")
    parser.add_argument("--markdown-output")
    parser.add_argument(
        "--strict",
        action="store_true",
        help="exit non-zero unless all five tiers and workloads pass",
    )
    args = parser.parse_args()

    matrix = load_manifest(args.manifest)
    normalized = []
    for path in _result_paths(args.results):
        document = json.loads(path.read_text(encoding="utf-8"))
        normalized.extend(
            normalize_result_document(matrix, document, str(path.resolve()))
        )
    report = evaluate_matrix(matrix, normalized)
    json_result = {
        "matrix_id": report.matrix_id,
        "global_status": report.global_status,
        "normalized_result_count": len(normalized),
        "rows": [asdict(row) for row in report.rows],
    }
    markdown = render_markdown(report)
    print(markdown, end="")
    if args.json_output:
        output = Path(args.json_output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(json_result, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
    if args.markdown_output:
        output = Path(args.markdown_output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(markdown, encoding="utf-8")
    return 1 if args.strict and report.global_status != "pass" else 0


if __name__ == "__main__":
    raise SystemExit(main())
