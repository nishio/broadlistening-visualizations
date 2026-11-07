"""チームみらい 2025 オープンデータを、広聴AI の公開用 artifact ``public_result.json`` の形に変換する。

このリポジトリの可視化は、どれも ``public_result.json``（kouchou-ai の公開契約、schema_version 1）を
入力にする。広聴AI で作ったどのレポートでも同じ可視化を描けるようにし、本体の公開ビューアへ
移すときに書き直しが要らないようにするため。契約は kouchou-ai の
``docs/development/public-result.md`` を参照。

    uv run scripts/team_mirai_to_public_result.py
    # -> data/team-mirai-2025/public_result.json
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import pandas as pd

QUESTION = "チームみらい 2025 参院選「しゃべれるマニフェスト」に届いた政策提案"
INTRO = (
    "2025 年参院選で、チームみらいの「しゃべれるマニフェスト」（AI インタビュアーが市民と対話して"
    "マニフェストへの変更提案を GitHub の PR として出す仕組み）に届いた 9,688 件の PR から、"
    "課題を抽出し、広聴AI で意見 21,348 件に分けて分析した結果。"
    "出典：チームみらい・しゃべれるマニフェスト参加者 / データセット "
    "nishiohirokazu/team-mirai-2025-embeddings（CC BY 4.0）"
)


def _clean(value):
    if value is None:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    return value


def convert(data_dir: Path) -> dict:
    args = pd.read_parquet(data_dir / "arguments.parquet")
    clusters = pd.read_parquet(data_dir / "clusters.parquet")
    overview_path = data_dir / "pipeline" / "kouchou-ai-overview.txt"
    overview = overview_path.read_text(encoding="utf-8").strip() if overview_path.exists() else ""

    arguments = []
    en_arguments = {}
    for row in args.itertuples(index=False):
        arguments.append(
            {
                "arg_id": row.id,
                "argument": row.argument,
                "comment_id": row.comment_id,
                "x": float(row.umap_x),
                "y": float(row.umap_y),
                "p": 0,
                "cluster_ids": ["0", row.cluster_level1, row.cluster_level2],
                "attributes": {"pr_number": int(row.pr_number)},
                "url": row.pr_url,
            }
        )
        if isinstance(row.argument_en, str) and row.argument_en:
            en_arguments[row.id] = row.argument_en

    out_clusters = []
    en_clusters = {}
    for row in clusters.sort_values(["level", "cluster_id"]).itertuples(index=False):
        out_clusters.append(
            {
                "level": int(row.level),
                "id": str(row.cluster_id),
                "label": row.label,
                "takeaway": row.takeaway or "",
                "value": int(row.num_arguments),
                "parent": _clean(row.parent) or "",
                "density_rank_percentile": _clean(row.density_rank_percentile),
            }
        )
        en_clusters[str(row.cluster_id)] = {"label": row.label_en or "", "takeaway": row.takeaway_en or ""}

    return {
        "schema_version": 1,
        "arguments": arguments,
        "clusters": out_clusters,
        "comments": {},
        "propertyMap": {"pr_number": {a["arg_id"]: a["attributes"]["pr_number"] for a in arguments}},
        "translations": {"en": {"clusters": en_clusters, "arguments": en_arguments}},
        "overview": overview,
        "comment_num": int(args.comment_id.nunique()),
        "config": {
            "question": QUESTION,
            "intro": INTRO,
            "model": "gpt-4o-mini",
            "provider": "openai",
            "enable_source_link": True,
            "enable_original_comments": False,
            "hierarchical_clustering": {"cluster_nums": [20, 400]},
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", type=Path, default=Path("data/team-mirai-2025"))
    parser.add_argument("--out", type=Path, default=None, help="既定は <data-dir>/public_result.json")
    args = parser.parse_args()
    out = args.out or args.data_dir / "public_result.json"
    result = convert(args.data_dir)
    out.write_text(json.dumps(result, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {out}: arguments={len(result['arguments'])} clusters={len(result['clusters'])}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
