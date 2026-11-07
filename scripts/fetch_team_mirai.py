"""チームみらい 2025「しゃべれるマニフェスト」オープンデータを Hugging Face から取ってくる。

https://huggingface.co/datasets/nishiohirokazu/team-mirai-2025-embeddings (CC BY 4.0)

    uv run scripts/fetch_team_mirai.py                 # 意見・クラスタ・PR・概要文
    uv run scripts/fetch_team_mirai.py --embeddings    # 広聴AI が UMAP に入れた 1536 次元の埋め込みも（約 120MB）
"""

from __future__ import annotations

import argparse
import sys
import urllib.request
from pathlib import Path

REPO = "nishiohirokazu/team-mirai-2025-embeddings"
BASE = f"https://huggingface.co/datasets/{REPO}/resolve/main/"
FILES = [
    "arguments.parquet",
    "clusters.parquet",
    "prs.parquet",
    "pipeline/kouchou-ai-overview.txt",
]
EMBEDDINGS = "kouchou-ai-embeddings-1536dim.parquet"


def fetch(name: str, out_dir: Path, force: bool) -> None:
    dest = out_dir / name
    if dest.exists() and not force:
        print(f"skip (exists): {dest}")
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    print(f"fetch: {BASE}{name}")
    with urllib.request.urlopen(BASE + name) as res, open(tmp, "wb") as f:
        while chunk := res.read(1 << 20):
            f.write(chunk)
    tmp.replace(dest)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out-dir", type=Path, default=Path("data/team-mirai-2025"))
    parser.add_argument("--embeddings", action="store_true", help="1536 次元の埋め込みも取る（約 120MB）")
    parser.add_argument("--force", action="store_true", help="既にあっても取り直す")
    args = parser.parse_args()
    names = FILES + ([EMBEDDINGS] if args.embeddings else [])
    for name in names:
        fetch(name, args.out_dir, args.force)
    return 0


if __name__ == "__main__":
    sys.exit(main())
