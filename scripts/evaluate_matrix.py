"""Score one model across several retrieval benchmarks and Matryoshka truncation dimensions.

``scripts/evaluate.py`` scores one model on one benchmark at one dimension, which means a full
picture of a checkpoint costs a dozen invocations and a dozen model loads. This driver loads the
model once per truncation dimension and writes every score to a single JSON file, so a release can
report the same set of measurements for every checkpoint instead of leaving gaps.

Reported figures are whatever :func:`morisien_embed.benchmark.evaluate` returns, which is the
``InformationRetrievalEvaluator`` the published numbers were produced with.
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import time
from pathlib import Path

from sentence_transformers import SentenceTransformer

from morisien_embed import benchmark

REPORTED = {
    "accuracy@1": "cosine_accuracy@1",
    "accuracy@10": "cosine_accuracy@10",
    "mrr@10": "cosine_mrr@10",
    "ndcg@10": "cosine_ndcg@10",
    "map@100": "cosine_map@100",
}


def pick(results: dict[str, float], suffix: str) -> float | None:
    return next((value for key, value in results.items() if key.endswith(suffix)), None)


def git_sha() -> str:
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True)
        return out.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def environment() -> dict[str, str]:
    import sentence_transformers
    import torch
    import transformers

    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "sentence-transformers": sentence_transformers.__version__,
        "transformers": transformers.__version__,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model")
    parser.add_argument("--revision", default=None, help="Hub revision of the model. Local paths ignore it.")
    parser.add_argument(
        "--data-dirs",
        type=Path,
        nargs="+",
        default=[Path("benchmark/data/eng")],
        help="benchmark directories, each scored at full dimension",
    )
    parser.add_argument(
        "--truncate-dims",
        type=int,
        nargs="*",
        default=[],
        help="extra Matryoshka dimensions, applied to the first data dir only",
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    scores: dict[str, dict[str, float | None]] = {}

    model = SentenceTransformer(args.model, revision=args.revision)
    # Renamed in sentence-transformers 5.7; the old name still works but warns.
    dimension = getattr(model, "get_embedding_dimension", None) or model.get_sentence_embedding_dimension
    full_dim = dimension()
    for data_dir in args.data_dirs:
        started = time.time()
        results = benchmark.evaluate(model, benchmark.load(data_dir), name=data_dir.name, batch_size=args.batch_size)
        scores[data_dir.name] = {label: pick(results, suffix) for label, suffix in REPORTED.items()}
        scores[data_dir.name]["seconds"] = round(time.time() - started, 1)
        print(f"[{data_dir.name}] ndcg@10 {scores[data_dir.name]['ndcg@10']:.4f}")

    # Truncation is a property of the loaded model, so each dimension needs its own load.
    truncation: dict[str, dict[str, float | None]] = {}
    if args.truncate_dims:
        primary = args.data_dirs[0]
        bench = benchmark.load(primary)
        truncation[str(full_dim)] = dict(scores[primary.name])
        for dim in args.truncate_dims:
            if dim == full_dim:
                continue
            truncated = SentenceTransformer(args.model, revision=args.revision, truncate_dim=dim)
            results = benchmark.evaluate(truncated, bench, name=f"{primary.name}-{dim}", batch_size=args.batch_size)
            truncation[str(dim)] = {label: pick(results, suffix) for label, suffix in REPORTED.items()}
            print(f"[{primary.name} @ {dim}d] ndcg@10 {truncation[str(dim)]['ndcg@10']:.4f}")

    report = {
        "model": args.model,
        "revision": args.revision,
        "embedding_dim": full_dim,
        "git": {"sha": git_sha()},
        "env": environment(),
        "benchmarks": {d.name: str(d) for d in args.data_dirs},
        "scores": scores,
    }
    if truncation:
        report["matryoshka_truncation"] = {"benchmark": args.data_dirs[0].name, "by_dim": truncation}

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"\nwrote {args.output}")


if __name__ == "__main__":
    main()
