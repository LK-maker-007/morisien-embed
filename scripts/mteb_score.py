"""Score a model on the MorisienMTBitextMining task in MTEB and record the per-subset F1."""

from __future__ import annotations

import argparse
import json
import platform
import statistics
from pathlib import Path

from sentence_transformers import SentenceTransformer

TASK = "MorisienMTBitextMining"


def load_task() -> object:
    import mteb

    try:
        return mteb.get_tasks(tasks=[TASK])
    except KeyError as error:
        raise SystemExit(
            f"{TASK} is not in mteb {mteb.__version__}. It was merged after 2.18, so install a "
            f"newer release, for example 'pip install mteb>=2.20'."
        ) from error


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("models", nargs="+", help="Hub ids or local paths")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    import mteb

    tasks = load_task()
    report: dict[str, object] = {
        "run": f"MTEB {TASK}, F1 per directional subset",
        "produced_by": "scripts/mteb_score.py",
        "task": {
            "name": TASK,
            "dataset": tasks[0].metadata.dataset["path"],
            "revision": tasks[0].metadata.dataset["revision"],
            "main_score": tasks[0].metadata.main_score,
        },
        "env": {
            "python": platform.python_version(),
            "mteb": mteb.__version__,
            "sentence-transformers": __import__("sentence_transformers").__version__,
            "transformers": __import__("transformers").__version__,
            "torch": __import__("torch").__version__,
        },
        "results": {},
    }

    for name in args.models:
        result = mteb.evaluate(SentenceTransformer(name), tasks, show_progress_bar=False)
        subsets: dict[str, float] = {}
        for task_result in result.task_results:
            for entries in task_result.scores.values():
                for entry in entries:
                    subsets[entry["hf_subset"]] = round(float(entry["f1"]), 4)
        report["results"][name] = {
            "by_subset": subsets,
            "mean_f1": round(statistics.fmean(subsets.values()), 4),
        }
        print(f"{name}: mean F1 {report['results'][name]['mean_f1']:.4f}  {subsets}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"\nwrote {args.output}")


if __name__ == "__main__":
    main()
