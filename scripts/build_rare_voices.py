"""「要約から落ちた声」用の事前計算：意見ごとに、似た意見がどれだけ少ないか（孤立度）を測る。

表札と要約は、まとまりの多数派を代表する。その反対側の、似た意見がほとんど無い声を候補として並べるため、
意見ごとに次を求める。重要かどうかは決めない。

- 孤立度：**同じ投稿の意見を除いて**、k 番目に近い意見との近さ。低いほど孤立している
  - 同じ投稿（束ねる属性が同じ意見。既定は comment_id）を除くのは、1 つの投稿から取り出した意見どうしは
    言い回しが似やすく、投稿が長いほど「似た意見がある」ように見えてしまうため。チームみらいのデータでは
    1 本の PR が複数の comment_id に分かれているので、--group-by pr_number で PR ごとに束ねる
  - 1 番目でなく k 番目（既定 10）にするのは、たまたま似た意見が 1 つあるだけで孤立していないことにしないため
  - --embeddings に (id, embedding) の parquet を渡すと、長さ 1 に正規化した埋め込みの cosine 類似度で測る。
    省くと public_result.json の 2 次元配置 x / y の距離で測る。ただし 2 次元では「配置の上で小さな島にいるか」を
    測ってしまう（チームみらいのデータでは、埋め込みで測った孤立した意見との重なりが 12% だった）
- いちばん近い「他の投稿の」意見とその近さ：本当に孤立しているかを読み手が確かめられるように
- 孤立した意見：孤立度の低い側から全体の --percent（既定 5）%。まとまり（上位・下位）ごとにその割合を出す。
  属性（arguments[].attributes）のうち値が 2〜8 通りのもの（このデータでは merged と author_kind）も、値ごとに割合を出す

生成物には孤立した意見だけを書く。本文・表札・英訳は書かない（表示側が public_result.json から arg_id で引く）。

    uv run scripts/build_rare_voices.py docs/data/team-mirai-2025/public_result.json \\
        --embeddings data/team-mirai-2025/kouchou-ai-embeddings-1536dim.parquet \\
        --group-by pr_number --out docs/rare/team-mirai-2025.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter, defaultdict
from datetime import date
from pathlib import Path

import numpy as np

QUANTILES = (1, 5, 10, 25, 50, 75, 90, 95, 99)


def load_vectors(public: dict, embeddings_path: Path | None) -> tuple[np.ndarray, str, str]:
    """意見の並び（public_result.json の arguments の順）に揃えたベクトルと、空間の名前・距離の種類を返す。"""
    ids = [a["arg_id"] for a in public["arguments"]]
    if embeddings_path is None:
        X = np.array([[a["x"], a["y"]] for a in public["arguments"]], dtype=np.float64)
        return X, "2d_layout", "euclidean"
    import pyarrow.parquet as pq

    table = pq.read_table(embeddings_path, columns=["id", "embedding"])
    emb_ids = table["id"].to_pylist()
    pos = {i: k for k, i in enumerate(emb_ids)}
    missing = [i for i in ids if i not in pos]
    if missing:
        raise SystemExit(f"埋め込みが無い意見が {len(missing)} 件あります（例: {missing[:3]}）")
    flat = np.asarray(table["embedding"].combine_chunks().flatten(), dtype=np.float32)
    E = flat.reshape(len(emb_ids), -1)[[pos[i] for i in ids]]
    E /= np.linalg.norm(E, axis=1, keepdims=True) + 1e-12  # cosine で測る
    return E, f"embeddings_{E.shape[1]}d", "cosine"


def group_codes(public: dict, group_by: str) -> tuple[np.ndarray, int]:
    """束ねる属性の値を整数に置き換える。属性が無い意見は、その意見の comment_id で束ねる（件数を返す）。"""
    keys, fallback = [], 0
    for a in public["arguments"]:
        if group_by == "comment_id":
            keys.append(("c", str(a["comment_id"])))
            continue
        v = (a.get("attributes") or {}).get(group_by)
        if v is None:
            fallback += 1
            keys.append(("c", str(a["comment_id"])))
        else:
            keys.append(("v", str(v)))
    _, codes = np.unique(np.array([f"{t}\t{v}" for t, v in keys]), return_inverse=True)
    return codes.astype(np.int64), fallback


def isolation(X: np.ndarray, groups: np.ndarray, k: int, metric: str, chunk: int = 1024):
    """各意見について、束ねる属性が違う意見の中での k 番目の近さと、いちばん近い意見・その近さを返す。

    近さは cosine なら類似度、euclidean なら距離に負号をつけたもの（どちらも大きいほど近い）。
    他の投稿の意見が k 件に満たない意見は -inf になる。
    """
    n = len(X)
    kth = np.empty(n)
    nn = np.empty(n, dtype=np.int64)
    nn_val = np.empty(n)
    if metric == "cosine":
        Xc = np.ascontiguousarray(X, dtype=np.float32)
    else:
        Xc = np.asarray(X, dtype=np.float64)
    # n - 1 で切り詰めると、意見が k 件より少ないレポートで黙って k を下げてしまう（meta の k と食い違う）。
    # n までにしておけば、他の投稿の意見が k 件に満たない行は k 番目が -inf になり、警告に回る
    kk = max(1, min(k, n))
    for s in range(0, n, chunk):
        e = min(n, s + chunk)
        if metric == "cosine":
            S = Xc[s:e] @ Xc.T
        else:
            D2 = np.zeros((e - s, n))
            for d in range(Xc.shape[1]):
                D2 += (Xc[s:e, d, None] - Xc[None, :, d]) ** 2
            S = -np.sqrt(D2)
        S[groups[s:e, None] == groups[None, :]] = -np.inf  # 自分自身と同じ投稿の意見を除く
        kth[s:e] = np.partition(S, -kk, axis=1)[:, -kk]
        j = S.argmax(axis=1)
        nn[s:e] = j
        nn_val[s:e] = S[np.arange(e - s), j]
    return kth, nn, nn_val


def attribute_shares(arguments: list[dict], rare: np.ndarray, group_by: str, max_values: int = 8) -> list[dict]:
    """値が 2〜max_values 通りの属性について、値ごとの意見数と孤立した意見の数。数値のスコアや自由記述は値が多いので外れる。"""
    total: dict[str, Counter] = defaultdict(Counter)
    hits: dict[str, Counter] = defaultdict(Counter)
    for i, a in enumerate(arguments):
        for key, v in (a.get("attributes") or {}).items():
            if key == group_by or v is None or isinstance(v, (float, dict, list)):
                continue
            total[key][json.dumps(v, ensure_ascii=False)] += 1
            hits[key][json.dumps(v, ensure_ascii=False)] += int(rare[i])
    out = []
    for key in sorted(total):
        if 2 <= len(total[key]) <= max_values:
            for v, c in total[key].most_common():
                out.append({"key": key, "value": json.loads(v), "n": c, "rare": hits[key][v]})
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("public_result", type=Path)
    parser.add_argument("--embeddings", type=Path, default=None, help="(id, embedding) の parquet。省略時は x/y の距離で測る")
    parser.add_argument("--group-by", default="comment_id", help="同じ投稿として束ねる属性（comment_id か attributes のキー）")
    parser.add_argument("--k", type=int, default=10, help="何番目に近い意見との近さを孤立度にするか")
    parser.add_argument("--percent", type=float, default=5.0, help="孤立した意見とみなす、孤立度の低い側からの割合（%%）")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.k < 1:
        parser.error("--k は 1 以上")
    if not 0 < args.percent <= 100:
        parser.error("--percent は 0 より大きく 100 以下")

    t0 = time.time()
    public = json.loads(args.public_result.read_text(encoding="utf-8"))
    if public.get("schema_version") != 1:
        print(f"warning: schema_version={public.get('schema_version')}（1 を想定）", file=sys.stderr)
    arguments = public["arguments"]
    n = len(arguments)
    X, space, metric = load_vectors(public, args.embeddings)
    groups, fallback = group_codes(public, args.group_by)
    if fallback:
        print(f"warning: {args.group_by} の無い意見 {fallback} 件は comment_id で束ねた", file=sys.stderr)
    sizes = np.bincount(groups)

    kth, nn, nn_val = isolation(X, groups, args.k, metric)
    short = int((~np.isfinite(kth)).sum())
    if short:
        print(f"warning: 他の投稿の意見が k={args.k} 件に満たない意見が {short} 件ある（孤立度を測れず、最も孤立した側に並ぶ。--k を小さくする）",
              file=sys.stderr)
    order = np.argsort(kth, kind="stable")  # 近さの低い順 = 孤立している順
    rank = np.empty(n, dtype=np.int64)
    rank[order] = np.arange(1, n + 1)
    n_rare = int(round(n * args.percent / 100))
    rare = rank <= n_rare
    show = (lambda v: v) if metric == "cosine" else (lambda v: -v)  # 書き出す値：類似度か距離

    def value(v: float):
        return round(float(show(v)), 4) if np.isfinite(v) else None  # 他の投稿の意見が k 件に満たないとき

    # まとまりごとの件数と、孤立した意見の件数（cluster_ids の 0 番目は全体なので数えない）
    total, hits = Counter(), Counter()
    for i, a in enumerate(arguments):
        for level, cid in enumerate(a["cluster_ids"]):
            if level == 0:
                continue
            total[cid] += 1
            hits[cid] += int(rare[i])

    # [arg_id, 孤立度の順位（1 がいちばん孤立）, k 番目の近さ, いちばん近い他の投稿の意見の arg_id, その近さ]
    voices = [
        [arguments[i]["arg_id"], int(rank[i]), value(kth[i]),
         arguments[int(nn[i])]["arg_id"] if np.isfinite(nn_val[i]) else None, value(nn_val[i])]
        for i in order[:n_rare]
    ]

    finite = kth[np.isfinite(kth)]
    q = np.percentile(show(finite), QUANTILES) if len(finite) else np.full(len(QUANTILES), np.nan)
    out = {
        "meta": {
            "built": date.today().isoformat(),
            "space": space,
            "metric": metric,
            "k": args.k,
            "group_by": args.group_by,
            "n_arguments": n,
            "n_groups": int(len(sizes)),
            "group_size_max": int(sizes.max()),
            "percent": args.percent,
            "n_rare": n_rare,
            "threshold": value(kth[order[n_rare - 1]]) if n_rare else None,
            "quantiles": {str(p): (round(float(v), 4) if np.isfinite(v) else None) for p, v in zip(QUANTILES, q)},
        },
        "voices": voices,
        "clusters": {cid: [total[cid], hits[cid]] for cid in total},
        "attributes": attribute_shares(arguments, rare, args.group_by),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")

    label = {c["id"]: c["label"] for c in public["clusters"]}
    print(f"wrote {args.out}: space={space} k={args.k} group_by={args.group_by} groups={len(sizes)} "
          f"voices={len(voices)} rare={n_rare} threshold={out['meta']['threshold']} ({time.time() - t0:.1f}s)")
    print("quantiles " + " ".join(f"p{p}={v:.3f}" for p, v in zip(QUANTILES, q)))
    print("-- 上位クラスタごとの、孤立した意見の割合（高い順）")
    upper = {a["cluster_ids"][1] for a in arguments if len(a["cluster_ids"]) > 1}
    for cid in sorted(upper, key=lambda c: -hits[c] / total[c]):
        print(f"  {cid:5} {100 * hits[cid] / total[cid]:5.1f}%  {hits[cid]:4} / {total[cid]:5}  {label.get(cid, '')}")
    print("-- 属性ごとの、孤立した意見の割合")
    for r in out["attributes"]:
        print(f"  {r['key']}={r['value']}  {100 * r['rare'] / r['n']:5.1f}%  {r['rare']:4} / {r['n']:5}")
    print("-- いちばん孤立した意見 15 件")
    for i in order[:15]:
        print(f"  {show(kth[i]):.3f} nn={show(nn_val[i]):.3f} | {arguments[i]['argument']}  ≒ {arguments[int(nn[i])]['argument']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
