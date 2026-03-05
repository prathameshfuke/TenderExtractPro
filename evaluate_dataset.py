from __future__ import annotations

import json
import statistics
import time
from pathlib import Path
from typing import Any, Dict, List

from tender_extraction.main import TenderExtractionPipeline


def mean_or_zero(values: List[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def summarize_result(result: Dict[str, Any]) -> Dict[str, Any]:
    specs = result.get("technical_specifications", []) or []
    scope = result.get("scope_of_work", {}) or {}

    confidences = [float(s.get("confidence", 0.0) or 0.0) for s in specs]

    return {
        "spec_count": len(specs),
        "deliverable_count": len(scope.get("deliverables", []) or []),
        "exclusion_count": len(scope.get("exclusions", []) or []),
        "location_count": len(scope.get("locations", []) or []),
        "reference_count": len(scope.get("references", []) or []),
        "high_confidence_specs": sum(1 for c in confidences if c >= 0.8),
        "medium_confidence_specs": sum(1 for c in confidences if 0.5 <= c < 0.8),
        "low_confidence_specs": sum(1 for c in confidences if c < 0.5),
        "avg_spec_confidence": round(mean_or_zero(confidences), 4),
        "accuracy_score": float(result.get("accuracy_score", 0.0) or 0.0),
    }


def write_markdown_report(report: Dict[str, Any], path: Path) -> None:
    docs = report["documents"]
    agg = report["aggregate"]

    lines = []
    lines.append("# TenderExtractPro Dataset Evaluation Report")
    lines.append("")
    lines.append(f"Generated at: `{report['generated_at']}`")
    lines.append(f"Runtime (total): `{report['runtime_seconds']:.1f}s`")
    lines.append("")
    lines.append("## Aggregate Metrics")
    lines.append("")
    lines.append(f"- Documents processed: **{agg['documents_processed']}**")
    lines.append(f"- Total extracted specs: **{agg['total_specs']}**")
    lines.append(f"- Mean specs/doc: **{agg['mean_specs_per_doc']:.2f}**")
    lines.append(f"- Mean grounding accuracy: **{agg['mean_accuracy_score']:.2f}%**")
    lines.append(f"- Mean spec confidence: **{agg['mean_spec_confidence']:.3f}**")
    lines.append(f"- Confidence distribution: **H {agg['total_high']} / M {agg['total_medium']} / L {agg['total_low']}**")
    lines.append("")
    lines.append("## Per-Document Results")
    lines.append("")
    lines.append("| Document | Time (s) | Specs | High | Medium | Low | Avg Conf | Accuracy % | Deliverables | Exclusions |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for d in docs:
        lines.append(
            f"| {d['filename']} | {d['elapsed_seconds']:.1f} | {d['spec_count']} | "
            f"{d['high_confidence_specs']} | {d['medium_confidence_specs']} | {d['low_confidence_specs']} | "
            f"{d['avg_spec_confidence']:.3f} | {d['accuracy_score']:.2f} | "
            f"{d['deliverable_count']} | {d['exclusion_count']} |"
        )

    lines.append("")
    lines.append("## Notes")
    lines.append("")
    lines.append("- Accuracy is grounding-based (how strongly extracted items match source chunks), not a human-labeled benchmark score.")
    lines.append("- Confidence values are post-validation scores in [0,1] derived from fuzzy grounding checks.")
    lines.append("- This report evaluates generalization across all available dataset files in the repository.")

    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    root = Path(__file__).resolve().parent
    dataset_dir = root / "dataset"
    outputs_dir = root / "outputs"
    outputs_dir.mkdir(exist_ok=True)

    docs = sorted([p for p in dataset_dir.iterdir() if p.suffix.lower() in {".pdf", ".docx", ".png", ".jpg", ".jpeg"}])
    if not docs:
        raise RuntimeError("No dataset files found.")

    pipeline = TenderExtractionPipeline(persist_dir=str(outputs_dir / "_qdrant_eval_storage"), force_reindex=True)
    all_rows: List[Dict[str, Any]] = []

    eval_start = time.time()
    for doc in docs:
        print(f"[EVAL] Processing {doc.name} ...")
        t0 = time.time()
        out_path = outputs_dir / f"eval_{doc.stem}.json"
        result = pipeline.run(str(doc), output_path=str(out_path))
        elapsed = time.time() - t0

        row = {
            "filename": doc.name,
            "output_file": out_path.name,
            "elapsed_seconds": round(elapsed, 2),
        }
        row.update(summarize_result(result))
        all_rows.append(row)

    runtime = time.time() - eval_start

    accuracy_values = [r["accuracy_score"] for r in all_rows]
    conf_values = [r["avg_spec_confidence"] for r in all_rows]
    spec_values = [r["spec_count"] for r in all_rows]

    aggregate = {
        "documents_processed": len(all_rows),
        "total_specs": sum(r["spec_count"] for r in all_rows),
        "mean_specs_per_doc": round(mean_or_zero(spec_values), 2),
        "median_specs_per_doc": round(statistics.median(spec_values), 2) if spec_values else 0.0,
        "mean_accuracy_score": round(mean_or_zero(accuracy_values), 2),
        "mean_spec_confidence": round(mean_or_zero(conf_values), 4),
        "total_high": sum(r["high_confidence_specs"] for r in all_rows),
        "total_medium": sum(r["medium_confidence_specs"] for r in all_rows),
        "total_low": sum(r["low_confidence_specs"] for r in all_rows),
    }

    report = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "runtime_seconds": round(runtime, 2),
        "documents": all_rows,
        "aggregate": aggregate,
    }

    json_path = outputs_dir / "dataset_eval_report.json"
    md_path = outputs_dir / "dataset_eval_report.md"
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    write_markdown_report(report, md_path)

    print(f"[EVAL] Wrote JSON report: {json_path}")
    print(f"[EVAL] Wrote Markdown report: {md_path}")


if __name__ == "__main__":
    main()
