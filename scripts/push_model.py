"""Push a trained model and the repository's model card to the Hugging Face Hub in one step.

``SentenceTransformer.push_to_hub`` alone regenerates an automatic card that headlines dev-split
metrics and carries no license or language metadata, so this script uploads the hand-written
``model_card.md`` as the final step — the Hub page always matches the reviewed card.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import HfApi
from sentence_transformers import SentenceTransformer


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model_dir", type=Path)
    parser.add_argument("repo_id")
    parser.add_argument("--card", type=Path, default=Path("model_card.md"))
    parser.add_argument("--public", action="store_true", help="create the repo public (default: private)")
    args = parser.parse_args()

    SentenceTransformer(str(args.model_dir)).push_to_hub(args.repo_id, private=not args.public)
    HfApi().upload_file(
        path_or_fileobj=str(args.card),
        path_in_repo="README.md",
        repo_id=args.repo_id,
        commit_message="Upload model card",
    )
    print(f"pushed {args.model_dir} + card -> https://huggingface.co/{args.repo_id}")


if __name__ == "__main__":
    main()
