"""Score a model with the xSIM++ error rate and a breakdown of which errors it makes.

xSIM++ reports an error rate, not a retrieval metric, and LASER's own script breaks those errors down
by the kind of perturbation the model fell for. Reporting nDCG instead makes a number that cannot be
compared against any other xSIM++ result, and throws away the diagnostic: a model that confuses
entities is failing differently from one that confuses numbers.

The released augmentation labels every distractor with the rule that produced it. There are three:
``entity_mention_replacement`` (38,855 of 44,033), ``number_replacement`` (3,262) and
``causality_alternation`` (1,916).

Reference: Chen et al., xSIM++: An Improved Proxy to Bitext Mining Performance for Low-Resource
Languages, ACL 2023. https://arxiv.org/abs/2306.12907
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from sentence_transformers import SentenceTransformer

from morisien_embed import benchmark, data
from morisien_embed.xsim import fetch_errtype


def score(model: SentenceTransformer, data_dir: Path, errtype: dict[str, dict[str, str]], batch_size: int) -> dict:
    """Error rate over the pool, with the misses grouped by perturbation rule.

    Args:
        model (`SentenceTransformer`): Encoder to score.
        data_dir (`Path`): Benchmark directory holding queries, corpus and qrels.
        errtype (`dict[str, dict[str, str]]`): xSIM++ map of augmented sentence to its rule.
        batch_size (`int`): Encoding batch size.

    Returns:
        `dict`: Error rate, error count, query count and a count per rule.
    """
    queries, corpus, qrels = benchmark.load(data_dir)
    qids, cids = list(queries), list(corpus)
    rule = {data.normalize(sentence): meta["errtype"] for sentence, meta in errtype.items()}

    q_emb = model.encode(
        [queries[q] for q in qids], batch_size=batch_size, convert_to_tensor=True, normalize_embeddings=True
    )
    c_emb = model.encode(
        [corpus[c] for c in cids], batch_size=batch_size, convert_to_tensor=True, normalize_embeddings=True
    )
    top = (q_emb @ c_emb.T).argmax(dim=1).tolist()

    index = {c: i for i, c in enumerate(cids)}
    errors: Counter[str] = Counter()
    for qid, retrieved in zip(qids, top, strict=True):
        if retrieved in {index[g] for g in qrels[qid]}:
            continue
        # A miss that is not one of the released distractors is some other passage in the pool.
        errors[rule.get(corpus[cids[retrieved]], "other_passage")] += 1

    total = sum(errors.values())
    return {
        "queries": len(qids),
        "errors": total,
        "error_rate": round(total / len(qids), 4),
        "by_rule": {k: errors[k] for k in sorted(errors)},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", nargs="+")
    parser.add_argument("--data-dir", type=Path, default=Path("benchmark/data/xsim-eng"))
    parser.add_argument("--cache-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--revision", default=None)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    errtype = fetch_errtype(args.cache_dir)
    results = {}
    for name in args.model:
        results[name] = score(
            SentenceTransformer(name, revision=args.revision), args.data_dir, errtype, args.batch_size
        )
        r = results[name]
        rules = "  ".join(f"{k} {v}" for k, v in r["by_rule"].items())
        print(f"{name}\n  error rate {r['error_rate']:.4f}  ({r['errors']}/{r['queries']})\n  {rules}")

    if args.output:
        args.output.write_text(json.dumps({"data_dir": str(args.data_dir), "results": results}, indent=2) + "\n")
        print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
