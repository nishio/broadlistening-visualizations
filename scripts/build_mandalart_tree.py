"""意見を「8 つ以下に割る」を、1 マスが意見 1 枚になるまで繰り返し、マンダラート用の木を作る。

広聴AI の既定の階層づくり（最細の段を k-means で切り、Ward で上へ併合する）では、親 1 つあたりの
子の数が揃わない（チームみらい 2025 のデータでは上位 20 クラスタの子の数が 1〜38）。マンダラートは
「中心 1 つ＋周り 8 マス」なので、ここでは上から順に k-means（k=8）で割り、子が 8 以下になった節は
意見そのものをマスに並べる。木の深さは枝ごとに変わってよい。

入力は広聴AI の公開用 artifact ``public_result.json``。``--embeddings`` に (id, embedding) の
parquet を渡すと高次元の埋め込みで割り、渡さなければ ``arguments[].x / y``（2 次元配置）で割る。

各節の表札は、LLM を使わず「その節の中心に最も近い意見」（medoid）をそのまま使う。

    uv run scripts/build_mandalart_tree.py data/team-mirai-2025/public_result.json \
        --embeddings data/team-mirai-2025/kouchou-ai-embeddings-1536dim.parquet \
        --out docs/mandalart/team-mirai-2025.json
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path

import numpy as np
from sklearn.cluster import KMeans

FANOUT = 8


def load_vectors(public: dict, embeddings_path: Path | None) -> tuple[np.ndarray, str]:
    ids = [a["arg_id"] for a in public["arguments"]]
    if embeddings_path is None:
        X = np.array([[a["x"], a["y"]] for a in public["arguments"]], dtype=np.float64)
        return X, "2d_layout"
    import pyarrow.parquet as pq

    table = pq.read_table(embeddings_path, columns=["id", "embedding"])
    emb_ids = table["id"].to_pylist()
    flat = np.asarray(table["embedding"].combine_chunks().flatten(), dtype=np.float32)
    E = flat.reshape(len(emb_ids), -1)
    pos = {i: k for k, i in enumerate(emb_ids)}
    missing = [i for i in ids if i not in pos]
    if missing:
        raise SystemExit(f"埋め込みが無い意見が {len(missing)} 件あります（例: {missing[:3]}）")
    X = E[[pos[i] for i in ids]].astype(np.float64)
    X /= np.linalg.norm(X, axis=1, keepdims=True) + 1e-12  # cosine で割る
    return X, f"embeddings_{X.shape[1]}d"


class Builder:
    def __init__(self, X: np.ndarray, seed: int, n_init: int):
        self.X = X
        self.seed = seed
        self.n_init = n_init
        self.nodes: list[dict] = []

    def medoid(self, idx: np.ndarray, avoid: int | None = None) -> int:
        center = self.X[idx].mean(axis=0)
        order = np.argsort(((self.X[idx] - center) ** 2).sum(axis=1))
        for k in order:
            if int(idx[k]) != avoid:
                return int(idx[k])
        return int(idx[order[0]])

    def split(self, idx: np.ndarray) -> list[np.ndarray]:
        km = KMeans(n_clusters=FANOUT, n_init=self.n_init, random_state=self.seed)
        labels = km.fit_predict(self.X[idx])
        groups = [idx[labels == k] for k in range(FANOUT)]
        groups = [g for g in groups if len(g)]
        if len(groups) < 2:  # 同じ点ばかりで割れないときは順番で等分する
            groups = [g for g in np.array_split(idx, FANOUT) if len(g)]
        return groups

    def build(self, idx: np.ndarray, depth: int = 0, parent_label: int | None = None) -> int:
        node_id = len(self.nodes)
        node = {"l": self.medoid(idx, avoid=parent_label), "s": int(len(idx)), "d": depth, "c": []}
        self.nodes.append(node)
        if len(idx) <= FANOUT:
            node["c"] = [f"a{int(i)}" for i in idx]
            return node_id
        children = []
        for g in sorted(self.split(idx), key=len, reverse=True):
            if len(g) == 1:
                children.append(f"a{int(g[0])}")
            else:
                children.append(f"n{self.build(g, depth + 1, parent_label=node['l'])}")
        node["c"] = children
        return node_id


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("public_result", type=Path)
    parser.add_argument("--embeddings", type=Path, default=None, help="(id, embedding) の parquet。省略時は x/y で割る")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-init", type=int, default=3)
    parser.add_argument("--title", default=None, help="中心のマスに出す題。省略時は config.question")
    args = parser.parse_args()

    public = json.loads(args.public_result.read_text(encoding="utf-8"))
    if public.get("schema_version") != 1:
        print(f"warning: schema_version={public.get('schema_version')}（1 を想定）", file=sys.stderr)
    X, space = load_vectors(public, args.embeddings)
    builder = Builder(X, seed=args.seed, n_init=args.n_init)
    builder.build(np.arange(len(X)))
    nodes = builder.nodes

    arguments = public["arguments"]
    en = (public.get("translations") or {}).get("en", {}).get("arguments") or {}
    config = public.get("config") or {}
    depths = [n["d"] for n in nodes]
    out = {
        "meta": {
            "title": args.title or config.get("question") or "",
            "intro": config.get("intro") or "",
            "space": space,
            "fanout": FANOUT,
            "label_method": "medoid",
            "seed": args.seed,
            "n_arguments": len(arguments),
            "n_nodes": len(nodes),
            "max_depth": max(depths),
            "built": date.today().isoformat(),
        },
        "args": {
            "text": [a["argument"] for a in arguments],
            "text_en": [en.get(a["arg_id"], "") for a in arguments] if en else None,
            "url": [a.get("url") or "" for a in arguments],
        },
        "nodes": nodes,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    hist = np.bincount(depths)
    print(f"wrote {args.out}: space={space} nodes={len(nodes)} max_depth={max(depths)} nodes_per_depth={hist.tolist()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
