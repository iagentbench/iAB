#!/usr/bin/env python3
"""
Generate global-sensemaking QA pairs from a run directory.

Reads from run_dir/curated_artifacts/ (curated_llm_package.json, entities_pruned.parquet),
runs generator + verifier models, and writes to run_dir/generated_artifacts/
(qa_pairs_all.json, qa_pairs_passed.json, generated_config.json, generate_config.yaml).
"""
import argparse
import concurrent.futures as cf
import json
import sys
import time
import traceback
from collections import Counter, deque
from datetime import datetime, timezone
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.curation.curate import find_run_dirs
from src.qa_generation.run import run_qa_generation


def _is_qa_ready_run_dir(run_dir: Path, curated_subdir: str = "curated_artifacts") -> bool:
    """
    True if run_dir has the curated artifacts QA generation expects.

    We still use `find_run_dirs(...)` for discovery (GraphRAG run layout), but QA generation
    reads from `curated_artifacts/`, so we skip discovered runs that haven't been curated yet.
    """
    curated_dir = Path(run_dir) / curated_subdir
    if not curated_dir.is_dir():
        return False
    required = [
        curated_dir / "curated_llm_package.json",
        curated_dir / "entities_pruned.parquet",
    ]
    return all(p.is_file() for p in required)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _fmt_seconds(seconds: float | None) -> str:
    if seconds is None:
        return "?"
    s = max(0.0, float(seconds))
    if s < 60:
        return f"{s:.1f}s"
    m = int(s // 60)
    rem = int(s % 60)
    if m < 60:
        return f"{m}m{rem:02d}s"
    h = int(m // 60)
    mm = int(m % 60)
    return f"{h}h{mm:02d}m"


def _batch_log(log_path: Path, message: str) -> None:
    """
    Append a timestamped line to a batch log file in the parent dir.
    This is intentionally lightweight (no logging config needed).
    """
    ts = _utc_now_iso()
    line = f"[{ts}] {message}"
    try:
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except OSError:
        pass


def _write_json(path: Path, obj: dict) -> None:
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(obj, f, indent=2, ensure_ascii=False)
    except OSError:
        pass


def _append_jsonl(path: Path, record: dict) -> None:
    try:
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except OSError:
        pass


def _truncate(s: str, max_chars: int = 4000) -> str:
    s = s or ""
    if len(s) <= max_chars:
        return s
    return s[: max_chars - 3] + "..."


def _qa_generation_worker(payload: dict) -> dict:
    """
    Process-pool worker for one run_dir.
    Returns a small dict so the parent can write logs/ledgers deterministically.
    """
    run_dir = Path(payload["run_dir"])
    t0 = time.perf_counter()
    try:
        config = run_qa_generation(
            run_dir,
            project_root=Path(payload["project_root"]),
            curated_subdir=payload["curated_subdir"],
            generated_subdir=payload["generated_subdir"],
            generation_model_id=payload["generation_model_id"],
            judge_model_ids=payload.get("judge_model_ids"),
            K_per_bundle=payload["K_per_bundle"],
            max_bundles=payload["max_bundles"],
            K_CONNECTOR=payload["K_CONNECTOR"],
            target_pass_count=payload["target_pass_count"],
            min_connectors_per_bundle=payload["min_connectors_per_bundle"],
            min_action_connectors_per_bundle=payload["min_action_connectors_per_bundle"],
            max_jaccard_overlap=payload["max_jaccard_overlap"],
            judge_max_attempts=payload["judge_max_attempts"],
            resume_if_completed=payload["resume_if_completed"],
        )
        elapsed = time.perf_counter() - t0
        return {
            "ok": True,
            "elapsed_s": round(elapsed, 4),
            "config": config,
            "exception": "",
            "traceback": "",
        }
    except Exception as e:
        elapsed = time.perf_counter() - t0
        return {
            "ok": False,
            "elapsed_s": round(elapsed, 4),
            "config": None,
            "exception": f"{type(e).__name__}: {e}",
            "traceback": traceback.format_exc(),
        }


def main():
    parser = argparse.ArgumentParser(
        description="Generate QA pairs from curated artifacts (output in generated_artifacts/)."
    )
    parser.add_argument(
        "run_dir",
        type=Path,
        help="Path to the run directory (e.g. output/New Pope chosen/20260116_164015), or parent dir when --recursive",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Treat path as parent directory; find all run dirs under it and generate QA for each (skips runs missing curated_artifacts/)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        metavar="N",
        help="Parallel workers for --recursive (processes; default: 1). Start with 2–3 to avoid model throttling.",
    )
    parser.add_argument(
        "--batch-ledger-name",
        type=str,
        default="qa_generation_batch_ledger.jsonl",
        help="Filename for batch ledger (JSONL) written under the parent dir when --recursive",
    )
    parser.add_argument(
        "--batch-log-name",
        type=str,
        default="qa_generation_batch.log",
        help="Filename for batch progress log written under the parent dir when --recursive",
    )
    parser.add_argument(
        "--batch-summary-name",
        type=str,
        default="qa_generation_batch_summary.json",
        help="Filename for batch summary JSON written under the parent dir when --recursive",
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=None,
        metavar="DIR",
        help=f"Project root for prompts (default: {project_root})",
    )
    parser.add_argument(
        "--generation-model",
        type=str,
        default="global.anthropic.claude-opus-4-5-20251101-v1:0",
        help="Bedrock model ID for QA generation",
    )
    parser.add_argument(
        "--judge-model",
        type=str,
        action="append",
        default=None,
        help="Bedrock model ID for an LLM judge (repeatable). Defaults to 3 independent judges.",
    )
    parser.add_argument(
        "--judge-max-attempts",
        type=int,
        default=3,
        help="Max attempts per judge per candidate on parse/JSON errors (default: 3)",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Do not resume; overwrite existing generated artifacts for this run_dir",
    )
    parser.add_argument(
        "--K-per-bundle",
        type=int,
        default=3,
        help="QA pairs to generate per bundle (default: 3)",
    )
    parser.add_argument(
        "--max-bundles",
        type=int,
        default=3,
        help="Max bundles to process (default: 3)",
    )
    parser.add_argument(
        "--K_CONNECTOR",
        type=int,
        default=12,
        help="Max connector relations per bundle (default: 12)",
    )
    parser.add_argument(
        "--target-pass-count",
        type=int,
        default=5,
        help="Stop after this many PASS items (default: 5)",
    )
    parser.add_argument(
        "--min-connectors-per-bundle",
        type=int,
        default=1,
        metavar="N",
        help="Min connector relations per bundle (default: 1; use 3+ for stricter quality)",
    )
    parser.add_argument(
        "--min-action-connectors-per-bundle",
        type=int,
        default=0,
        metavar="N",
        help="Min action-like (non-identity) connectors per bundle (default: 0; use 1+ for stricter quality)",
    )
    parser.add_argument(
        "--max-jaccard-overlap",
        type=float,
        default=0.8,
        metavar="F",
        help="Max Jaccard overlap on topic_signature between communities (default: 0.8; use 1.0 to disable overlap filter)",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Print progress and config",
    )
    args = parser.parse_args()

    proot = args.project_root.resolve() if args.project_root else project_root
    if not proot.is_dir():
        print(f"Error: project root not found: {proot}", file=sys.stderr)
        return 1

    path = args.run_dir.resolve()
    if not path.is_dir():
        print(f"Error: directory not found: {path}", file=sys.stderr)
        return 1

    # Batch mode: discover all GraphRAG run dirs under a parent directory (recursive).
    if args.recursive:
        run_dirs = find_run_dirs(path)
        if not run_dirs:
            print(f"No run directories found under {path}", file=sys.stderr)
            return 1
        if args.verbose:
            print(f"Found {len(run_dirs)} run directory(ies) under {path}")

        batch_log_path = path / str(args.batch_log_name)
        batch_ledger_path = path / str(args.batch_ledger_name)
        batch_summary_path = path / str(args.batch_summary_name)
        _batch_log(
            batch_log_path,
            f"Starting batch QA generation. parent_dir={path} runs_total={len(run_dirs)} workers={args.workers}",
        )

        # Rolling mean for ETA (exclude quick skips/resume to avoid underestimation).
        recent_durations = deque(maxlen=10)
        outcomes = Counter()
        batch_t0 = time.perf_counter()

        # Keep a tiny copy of the effective CLI args for traceability in the ledger.
        args_for_ledger = {
            "generation_model": args.generation_model,
            "judge_model": args.judge_model,
            "K_per_bundle": args.K_per_bundle,
            "max_bundles": args.max_bundles,
            "K_CONNECTOR": args.K_CONNECTOR,
            "target_pass_count": args.target_pass_count,
            "min_connectors_per_bundle": args.min_connectors_per_bundle,
            "min_action_connectors_per_bundle": args.min_action_connectors_per_bundle,
            "max_jaccard_overlap": args.max_jaccard_overlap,
            "judge_max_attempts": args.judge_max_attempts,
            "resume_if_completed": (not args.no_resume),
        }

        def _eta_str(done_count: int) -> str:
            if not recent_durations:
                return "eta=?"
            avg = sum(recent_durations) / len(recent_durations)
            remaining = max(0, len(run_dirs) - done_count)
            return f"eta={_fmt_seconds(avg * remaining)}"

        # Workers <= 1: keep sequential behavior (deterministic, simplest).
        if args.workers <= 1:
            for idx, run_dir in enumerate(run_dirs, start=1):
                run_started_at = _utc_now_iso()
                t0 = time.perf_counter()

                # Skip runs without curated artifacts required for QA packet construction.
                if not _is_qa_ready_run_dir(run_dir, curated_subdir="curated_artifacts"):
                    outcomes["skipped_missing_curated"] += 1
                    msg = f"[{idx}/{len(run_dirs)}] skipped_missing_curated run_dir={run_dir} {_eta_str(idx)}"
                    print(msg)
                    _batch_log(batch_log_path, msg)
                    _append_jsonl(
                        batch_ledger_path,
                        {
                            "index": idx,
                            "total": len(run_dirs),
                            "run_dir": str(run_dir),
                            "outcome": "skipped_missing_curated",
                            "started_at_utc": run_started_at,
                            "ended_at_utc": _utc_now_iso(),
                            "elapsed_s": round(time.perf_counter() - t0, 4),
                            "args": args_for_ledger,
                        },
                    )
                    continue

                generated_dir = run_dir / "generated_artifacts"
                resume_marker_pre = (
                    (not args.no_resume)
                    and (generated_dir / "generated_config.json").is_file()
                    and (generated_dir / "qa_pairs_passed.json").is_file()
                )

                config = None
                outcome = None
                exc = ""
                tb = ""
                try:
                    config = run_qa_generation(
                        run_dir,
                        project_root=proot,
                        curated_subdir="curated_artifacts",
                        generated_subdir="generated_artifacts",
                        generation_model_id=args.generation_model,
                        judge_model_ids=args.judge_model,
                        K_per_bundle=args.K_per_bundle,
                        max_bundles=args.max_bundles,
                        K_CONNECTOR=args.K_CONNECTOR,
                        target_pass_count=args.target_pass_count,
                        min_connectors_per_bundle=args.min_connectors_per_bundle,
                        min_action_connectors_per_bundle=args.min_action_connectors_per_bundle,
                        max_jaccard_overlap=args.max_jaccard_overlap,
                        judge_max_attempts=args.judge_max_attempts,
                        resume_if_completed=not args.no_resume,
                    )

                    status = (config or {}).get("status", "")
                    packets_count = int((config or {}).get("packets_count", 0) or 0)
                    errors = (config or {}).get("errors") or []
                    resumed = bool(resume_marker_pre and status == "completed" and not errors)

                    if resumed:
                        outcome = "skipped_completed_resume"
                    elif errors or status == "completed_with_errors":
                        outcome = "completed_with_errors"
                    elif packets_count == 0:
                        outcome = "completed_no_packets"
                    else:
                        outcome = "completed_with_packets"

                except Exception as e:
                    exc = f"{type(e).__name__}: {e}"
                    tb = traceback.format_exc()
                    outcome = "failed_exception"

                elapsed = time.perf_counter() - t0
                if outcome in {"completed_with_packets", "completed_no_packets", "completed_with_errors", "failed_exception"}:
                    recent_durations.append(elapsed)

                outcomes[outcome] += 1
                run_ended_at = _utc_now_iso()

                packets = config.get("packets_count") if isinstance(config, dict) else None
                passed = config.get("qa_pairs_passed_count") if isinstance(config, dict) else None
                suffix = []
                if packets is not None:
                    suffix.append(f"packets={packets}")
                if passed is not None:
                    suffix.append(f"passed={passed}")
                msg = (
                    f"[{idx}/{len(run_dirs)}] {outcome} run_dir={run_dir} "
                    f"elapsed={_fmt_seconds(elapsed)} {_eta_str(idx)} "
                    + (" ".join(suffix) if suffix else "")
                ).strip()
                print(msg)
                _batch_log(batch_log_path, msg)
                if outcome == "failed_exception":
                    print(f"  error: {exc}", file=sys.stderr)
                    _batch_log(batch_log_path, f"  error: {exc}")
                    if args.verbose and tb:
                        print(tb, file=sys.stderr)

                _append_jsonl(
                    batch_ledger_path,
                    {
                        "index": idx,
                        "total": len(run_dirs),
                        "run_dir": str(run_dir),
                        "outcome": outcome,
                        "started_at_utc": run_started_at,
                        "ended_at_utc": run_ended_at,
                        "elapsed_s": round(elapsed, 4),
                        "resume_marker_pre": bool(resume_marker_pre),
                        "config": config if isinstance(config, dict) else None,
                        "exception": _truncate(exc),
                        "traceback": _truncate(tb),
                        "args": args_for_ledger,
                    },
                )
        else:
            # Parallel mode: process run dirs concurrently; parent remains the single writer for logs/ledgers.
            max_workers = max(1, int(args.workers))
            _batch_log(batch_log_path, f"Parallel mode enabled. max_workers={max_workers}")

            to_submit = []
            # Ledger entries for immediate skips keep discovered order.
            for idx, run_dir in enumerate(run_dirs, start=1):
                if not _is_qa_ready_run_dir(run_dir, curated_subdir="curated_artifacts"):
                    outcomes["skipped_missing_curated"] += 1
                    msg = f"[{idx}/{len(run_dirs)}] skipped_missing_curated run_dir={run_dir} {_eta_str(idx)}"
                    print(msg)
                    _batch_log(batch_log_path, msg)
                    _append_jsonl(
                        batch_ledger_path,
                        {
                            "index": idx,
                            "total": len(run_dirs),
                            "run_dir": str(run_dir),
                            "outcome": "skipped_missing_curated",
                            "started_at_utc": _utc_now_iso(),
                            "ended_at_utc": _utc_now_iso(),
                            "elapsed_s": 0.0,
                            "args": args_for_ledger,
                        },
                    )
                    continue

                generated_dir = run_dir / "generated_artifacts"
                resume_marker_pre = (
                    (not args.no_resume)
                    and (generated_dir / "generated_config.json").is_file()
                    and (generated_dir / "qa_pairs_passed.json").is_file()
                )
                to_submit.append(
                    {
                        "index": idx,
                        "run_dir": run_dir,
                        "resume_marker_pre": bool(resume_marker_pre),
                        "started_at_utc": _utc_now_iso(),
                    }
                )

            worker_payload_base = {
                "project_root": str(proot),
                "curated_subdir": "curated_artifacts",
                "generated_subdir": "generated_artifacts",
                "generation_model_id": args.generation_model,
                "judge_model_ids": args.judge_model,
                "K_per_bundle": args.K_per_bundle,
                "max_bundles": args.max_bundles,
                "K_CONNECTOR": args.K_CONNECTOR,
                "target_pass_count": args.target_pass_count,
                "min_connectors_per_bundle": args.min_connectors_per_bundle,
                "min_action_connectors_per_bundle": args.min_action_connectors_per_bundle,
                "max_jaccard_overlap": args.max_jaccard_overlap,
                "judge_max_attempts": args.judge_max_attempts,
                "resume_if_completed": (not args.no_resume),
            }

            done_count = int(outcomes.get("skipped_missing_curated", 0))
            futures: dict[cf.Future, dict] = {}
            with cf.ProcessPoolExecutor(max_workers=max_workers) as ex:
                for meta in to_submit:
                    payload = dict(worker_payload_base)
                    payload["run_dir"] = str(meta["run_dir"])
                    fut = ex.submit(_qa_generation_worker, payload)
                    futures[fut] = meta

                for fut in cf.as_completed(futures):
                    meta = futures[fut]
                    idx = meta["index"]
                    run_dir = meta["run_dir"]
                    run_started_at = meta["started_at_utc"]
                    resume_marker_pre = bool(meta["resume_marker_pre"])

                    result = None
                    try:
                        result = fut.result()
                    except Exception as e:
                        # Should be rare because worker catches, but keep fail-closed.
                        result = {
                            "ok": False,
                            "elapsed_s": 0.0,
                            "config": None,
                            "exception": f"{type(e).__name__}: {e}",
                            "traceback": traceback.format_exc(),
                        }

                    elapsed_s = float(result.get("elapsed_s") or 0.0)
                    config = result.get("config") if isinstance(result.get("config"), dict) else None
                    exc = str(result.get("exception") or "")
                    tb = str(result.get("traceback") or "")

                    outcome = None
                    if not result.get("ok"):
                        outcome = "failed_exception"
                    else:
                        status = (config or {}).get("status", "")
                        packets_count = int((config or {}).get("packets_count", 0) or 0)
                        errors = (config or {}).get("errors") or []
                        resumed = bool(resume_marker_pre and status == "completed" and not errors)
                        if resumed:
                            outcome = "skipped_completed_resume"
                        elif errors or status == "completed_with_errors":
                            outcome = "completed_with_errors"
                        elif packets_count == 0:
                            outcome = "completed_no_packets"
                        else:
                            outcome = "completed_with_packets"

                    # ETA excludes quick skips/resumes (same idea as sequential).
                    if outcome in {"completed_with_packets", "completed_no_packets", "completed_with_errors", "failed_exception"}:
                        recent_durations.append(elapsed_s)

                    outcomes[outcome] += 1
                    done_count += 1
                    run_ended_at = _utc_now_iso()

                    packets = (config or {}).get("packets_count") if config else None
                    passed = (config or {}).get("qa_pairs_passed_count") if config else None
                    suffix = []
                    if packets is not None:
                        suffix.append(f"packets={packets}")
                    if passed is not None:
                        suffix.append(f"passed={passed}")
                    msg = (
                        f"[{done_count}/{len(run_dirs)}] {outcome} run_dir={run_dir} "
                        f"elapsed={_fmt_seconds(elapsed_s)} {_eta_str(done_count)} idx={idx} "
                        + (" ".join(suffix) if suffix else "")
                    ).strip()
                    print(msg)
                    _batch_log(batch_log_path, msg)
                    if outcome == "failed_exception":
                        print(f"  error: {exc}", file=sys.stderr)
                        _batch_log(batch_log_path, f"  error: {exc}")
                        if args.verbose and tb:
                            print(tb, file=sys.stderr)

                    _append_jsonl(
                        batch_ledger_path,
                        {
                            "index": idx,
                            "total": len(run_dirs),
                            "run_dir": str(run_dir),
                            "outcome": outcome,
                            "started_at_utc": run_started_at,
                            "ended_at_utc": run_ended_at,
                            "elapsed_s": round(elapsed_s, 4),
                            "resume_marker_pre": bool(resume_marker_pre),
                            "config": config,
                            "exception": _truncate(exc),
                            "traceback": _truncate(tb),
                            "args": args_for_ledger,
                        },
                    )

        batch_elapsed = time.perf_counter() - batch_t0
        summary = {
            "parent_dir": str(path),
            "runs_total": len(run_dirs),
            "outcomes": dict(outcomes),
            "elapsed_s": round(batch_elapsed, 4),
            "elapsed_human": _fmt_seconds(batch_elapsed),
            "batch_log": str(batch_log_path),
            "batch_ledger_jsonl": str(batch_ledger_path),
            "created_at_utc": _utc_now_iso(),
            "args": args_for_ledger,
        }
        _write_json(batch_summary_path, summary)
        _batch_log(batch_log_path, f"Batch finished. summary={batch_summary_path} elapsed={_fmt_seconds(batch_elapsed)}")

        print(
            "Batch results: "
            f"completed_with_packets={outcomes.get('completed_with_packets', 0)} "
            f"completed_no_packets={outcomes.get('completed_no_packets', 0)} "
            f"completed_with_errors={outcomes.get('completed_with_errors', 0)} "
            f"skipped_missing_curated={outcomes.get('skipped_missing_curated', 0)} "
            f"skipped_completed_resume={outcomes.get('skipped_completed_resume', 0)} "
            f"failed_exception={outcomes.get('failed_exception', 0)} "
            f"(elapsed={_fmt_seconds(batch_elapsed)}). "
            f"Ledger: {batch_ledger_path} Summary: {batch_summary_path}"
        )

        failed = int(outcomes.get("failed_exception", 0))
        return 0 if failed == 0 else 1

    # Single run mode.
    run_dir = path
    try:
        config = run_qa_generation(
            run_dir,
            project_root=proot,
            curated_subdir="curated_artifacts",
            generated_subdir="generated_artifacts",
            generation_model_id=args.generation_model,
            judge_model_ids=args.judge_model,
            K_per_bundle=args.K_per_bundle,
            max_bundles=args.max_bundles,
            K_CONNECTOR=args.K_CONNECTOR,
            target_pass_count=args.target_pass_count,
            min_connectors_per_bundle=args.min_connectors_per_bundle,
            min_action_connectors_per_bundle=args.min_action_connectors_per_bundle,
            max_jaccard_overlap=args.max_jaccard_overlap,
            judge_max_attempts=args.judge_max_attempts,
            resume_if_completed=not args.no_resume,
        )
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1

    packets_count = config.get("packets_count", 0)
    qa_passed_count = config.get("qa_pairs_passed_count", 0)

    if packets_count == 0:
        min_conn = config.get("min_connectors_per_bundle", "?")
        min_act = config.get("min_action_connectors_per_bundle", "?")
        max_jacc = config.get("max_jaccard_overlap", "?")
        print(
            "No QA packets built (no connector-linked community pairs passed filters). "
            f"Effective filters: min_connectors_per_bundle={min_conn}, min_action_connectors_per_bundle={min_act}, "
            f"max_jaccard_overlap={max_jacc}. "
            "Relax with e.g. --min-connectors-per-bundle 1 --min-action-connectors-per-bundle 0 "
            "or raise --max-jaccard-overlap (e.g. 1.0).",
            file=sys.stderr,
        )
    else:
        print(f"Built {packets_count} QA packet(s); wrote {qa_passed_count} PASS QA pair(s).")

    if args.verbose:
        print("Generated artifacts written to:", config["generated_artifacts_dir"])
        print("  qa_pairs_all:", config["paths"]["qa_pairs_all"])
        print("  qa_pairs_passed:", config["paths"]["qa_pairs_passed"])
        print("  packets:", config["packets_count"], "passed:", config.get("qa_pairs_passed_count", 0))
        if config.get("errors"):
            for err in config["errors"]:
                print("  error:", err, file=sys.stderr)

    print("OK generated_artifacts (qa_pairs_all.json, qa_pairs_passed.json, generated_config.json, generate_config.yaml)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
