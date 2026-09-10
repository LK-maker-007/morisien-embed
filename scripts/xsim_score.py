"""Score a model with the xSIM++ error rate and a breakdown of which errors it makes."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

from morisien_embed import benchmark, data
from morisien_embed.xsim import fetch_errtype

MARGINS = ("ratio", "distance", "absolute")
CHUNK = 4096  # passages per pass; caps peak memory at queries x CHUNK floats


def retrieve_many(q_emb: torch.Tensor, c_emb: torch.Tensor, margins: tuple[str, ...], k: int) -> dict[str, list[int]]:
    cos_xy = torch.full((q_emb.size(0), k), -torch.inf)
    idx_xy = torch.zeros((q_emb.size(0), k), dtype=torch.long)
    avg_yx = torch.empty(c_emb.size(0))

    for start in range(0, c_emb.size(0), CHUNK):
        chunk = c_emb[start : start + CHUNK]
        sims = q_emb @ chunk.T
        # Each passage averages its k best queries. Fewer queries than k would make this ill-defined.
        avg_yx[start : start + chunk.size(0)] = sims.topk(min(k, sims.size(0)), dim=0).values.mean(dim=0)
        # Merge this chunk's best into the running best, keeping global passage indices.
        merged_cos = torch.cat([cos_xy, sims], dim=1)
        merged_idx = torch.cat([idx_xy, torch.arange(start, start + chunk.size(0)).expand(q_emb.size(0), -1)], dim=1)
        cos_xy, order = merged_cos.topk(k, dim=1)
        idx_xy = merged_idx.gather(1, order)

    avg_xy = cos_xy.mean(dim=1)
    denominator = (avg_xy.unsqueeze(1) + avg_yx[idx_xy]) / 2
    rows = torch.arange(idx_xy.size(0))

    aligned: dict[str, list[int]] = {}
    for margin in margins:
        if margin == "absolute":
            aligned[margin] = idx_xy[:, 0].tolist()
            continue
        scores = cos_xy / denominator if margin == "ratio" else cos_xy - denominator
        aligned[margin] = idx_xy[rows, scores.argmax(dim=1)].tolist()
    return aligned


def retrieve(q_emb: torch.Tensor, c_emb: torch.Tensor, margin: str, k: int) -> list[int]:
    return retrieve_many(q_emb, c_emb, (margin,), k)[margin]


def score(
    model: SentenceTransformer,
    data_dir: Path,
    errtype: dict[str, dict[str, str]],
    batch_size: int,
    *,
    margins: tuple[str, ...] = ("ratio",),
    k: int = 4,
    per_query: Path | None = None,
    label: str = "model",
) -> dict:
    queries, corpus, qrels = benchmark.load(data_dir)
    qids, cids = list(queries), list(corpus)
    rule = {data.normalize(sentence): meta["errtype"] for sentence, meta in errtype.items()}

    q_emb = model.encode(
        [queries[q] for q in qids], batch_size=batch_size, convert_to_tensor=True, normalize_embeddings=True
    )
    c_emb = model.encode(
        [corpus[c] for c in cids], batch_size=batch_size, convert_to_tensor=True, normalize_embeddings=True
    )
    index = {c: i for i, c in enumerate(cids)}
    gold = {qid: {index[g] for g in qrels[qid]} for qid in qids}

    aligned = retrieve_many(q_emb, c_emb, margins, k)
    report: dict[str, dict] = {}
    for margin in margins:
        errors: Counter[str] = Counter()
        hits = np.zeros(len(qids), dtype=np.int64)
        for position, (qid, retrieved) in enumerate(zip(qids, aligned[margin], strict=True)):
            if retrieved in gold[qid]:
                hits[position] = 1
                continue
            # A miss that is not one of the released distractors is some other passage in the pool.
            errors[rule.get(corpus[cids[retrieved]], "other_passage")] += 1
        if per_query is not None:
            per_query.mkdir(parents=True, exist_ok=True)
            np.savez(per_query / f"{label}-{margin}.npz", hits=hits, qids=np.array(qids))
        total = sum(errors.values())
        assert total == len(qids) - int(hits.sum()), "error count and hit vector disagree"
        report[margin] = {
            "k": k if margin != "absolute" else None,
            "queries": len(qids),
            "errors": total,
            "error_rate": round(total / len(qids), 4),
            "by_rule": {name: errors[name] for name in sorted(errors)},
        }
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", nargs="+")
    parser.add_argument("--data-dir", type=Path, default=Path("benchmark/data/xsim-eng"))
    parser.add_argument("--cache-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--revision", default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument(
        "--margin",
        nargs="+",
        choices=MARGINS,
        default=["ratio"],
        help="scoring function; 'ratio' is what LASER's xsim.py defaults to and what published xSIM++ numbers use",
    )
    parser.add_argument("--k", type=int, default=4, help="neighbours the margin reranks, ignored for 'absolute'")
    parser.add_argument(
        "--per-query", type=Path, default=None, help="directory for per-query hit vectors, needed for paired tests"
    )
    args = parser.parse_args()

    errtype = fetch_errtype(args.cache_dir)
    results = {}
    for name in args.model:
        results[name] = score(
            SentenceTransformer(name, revision=args.revision),
            args.data_dir,
            errtype,
            args.batch_size,
            margins=tuple(args.margin),
            k=args.k,
            per_query=args.per_query,
            label=name.replace("/", "_"),
        )
        print(name)
        for margin, r in results[name].items():
            rules = "  ".join(f"{rule} {count}" for rule, count in r["by_rule"].items())
            print(f"  [{margin:8}] error rate {r['error_rate']:.4f}  ({r['errors']}/{r['queries']})   {rules}")

    if args.output:
        report = {"data_dir": str(args.data_dir), "margin": args.margin, "k": args.k, "results": results}
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
