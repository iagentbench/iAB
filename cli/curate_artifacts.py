#!/usr/bin/env python3
"""
Curate artifacts from a GraphRAG run directory.

Creates curated_artifacts/ with:
- curated_llm_package.json  (indexable: meta, connector_relations, communities)
- entities_pruned.parquet  (cleaned entities)
- relationships_pruned.parquet (cleaned relationships, is_connector column)
- curated_config.json (K_CONNECTOR, M_FINDINGS, connector count, paths for traceability/plotter)
"""
import argparse
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.curation.curate import run_curate, find_run_dirs, resolve_input_dir_for_run


def main():
    parser = argparse.ArgumentParser(
        description="Curate artifacts from an iAgentBench run directory (curated_artifacts/)."
    )
    parser.add_argument(
        "run_dir",
        type=Path,
        help="Path to the run directory (e.g. output/MyKeyword/20260116_120000), or parent dir when --recursive",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Treat path as parent directory; find all run dirs under it and curate each",
    )
    parser.add_argument(
        "--input-root",
        type=Path,
        default=None,
        metavar="DIR",
        help="Root of input dirs for URL resolution (e.g. input/2025_seeds). For each run, input_dir = input_root/keywords/<query> or input_root/<query>. Use with --recursive when output is under a batch (e.g. output/2025_test_subset) and inputs are under input/2025_seeds.",
    )
    parser.add_argument(
        "--out-subdir",
        type=str,
        default="curated_artifacts",
        help="Output subdirectory name (default: curated_artifacts)",
    )
    parser.add_argument(
        "--K_CONNECTOR",
        type=int,
        default=12,
        help="Max connector relations to include in LLM package (default: 12)",
    )
    parser.add_argument(
        "--M_FINDINGS",
        type=int,
        default=5,
        help="Max findings per community (default: 5)",
    )
    parser.add_argument(
        "--include-communities-without-findings",
        action="store_true",
        help="Include communities that have no findings in the package",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Print written paths and config summary",
    )
    args = parser.parse_args()

    path = args.run_dir.resolve()
    if not path.is_dir():
        print(f"Error: directory not found: {path}", file=sys.stderr)
        return 1

    input_root = args.input_root.resolve() if args.input_root else None
    if input_root is not None and not input_root.is_dir():
        print(f"Error: input-root directory not found: {input_root}", file=sys.stderr)
        return 1

    if args.recursive:
        run_dirs = find_run_dirs(path)
        if not run_dirs:
            print(f"No run directories found under {path}", file=sys.stderr)
            return 1
        if args.verbose:
            print(f"Found {len(run_dirs)} run directory(ies) under {path}")
        failed = 0
        for run_dir in run_dirs:
            input_dir = resolve_input_dir_for_run(input_root, run_dir) if input_root else None
            try:
                config = run_curate(
                    run_dir=run_dir,
                    out_subdir=args.out_subdir,
                    K_CONNECTOR=args.K_CONNECTOR,
                    M_FINDINGS=args.M_FINDINGS,
                    INCLUDE_COMMUNITIES_WITHOUT_FINDINGS=args.include_communities_without_findings,
                    input_dir=input_dir,
                )
                if args.verbose:
                    print("OK", run_dir, "->", config["curated_artifacts_dir"])
            except Exception as e:
                failed += 1
                print(f"Error curating {run_dir}: {e}", file=sys.stderr)
                if args.verbose:
                    import traceback
                    traceback.print_exc()
        print(f"OK curated_artifacts: {len(run_dirs) - failed}/{len(run_dirs)} run(s) succeeded")
        return 0 if failed == 0 else 1

    input_dir = resolve_input_dir_for_run(input_root, path) if input_root else None
    try:
        config = run_curate(
            run_dir=path,
            out_subdir=args.out_subdir,
            K_CONNECTOR=args.K_CONNECTOR,
            M_FINDINGS=args.M_FINDINGS,
            INCLUDE_COMMUNITIES_WITHOUT_FINDINGS=args.include_communities_without_findings,
            input_dir=input_dir,
        )
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1

    if args.verbose:
        print("Curated artifacts written to:", config["curated_artifacts_dir"])
        for key, path in config["paths"].items():
            print(f"  {key}: {path}")
        print("Config: K_CONNECTOR={}, M_FINDINGS={}, connector_count={}".format(
            config["K_CONNECTOR"], config["M_FINDINGS"], config["connector_count"]
        ))

    print("OK curated_artifacts (curated_llm_package.json, entities_pruned.parquet, relationships_pruned.parquet, curated_config.json)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
