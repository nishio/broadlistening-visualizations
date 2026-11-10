"""同じ投稿の中で一緒に書かれた下位クラスタの組のうち、偶然では説明できないものを選ぶ（つながり線の事前計算）。

散布図は「文が似ているか」で意見を置く。こちらは「同じ投稿の中で一緒に書かれたか」で下位クラスタどうしを結ぶ。
地図の上で遠いのに一緒に書かれる組は、散布図の裂け目の候補になる（文も遠ければ、別の話題を一緒に書いているだけかもしれない）。

- 束ねる単位：既定は意見の ``comment_id``。``--group-by <属性名>`` を渡すと ``arguments[].attributes`` の
  その属性で束ねる（例：チームみらいのデータは 1 本の PR を課題ごとの comment に分けているので ``pr_number``）
- 単位ごとに、含まれる下位クラスタの集合を作る。組 (x, y) ごとに、両方を含む単位の数 k と、
  偶然なら見込まれる数 λ を出し、ポアソン分布の P(X ≥ k) を p 値にする。lift = k / λ
- λ は既定で層別（``--null strat``）：含む下位クラスタの数 s が同じ単位どうしで
  n_x^s · n_y^s · (s−1) / (s · N_s) を出し（n_x^s は x を含む単位の数、N_s は単位の数）、s について足す。
  まとまりを 1 つしか含まない単位は組を作れず、多く含む長い単位ほど組を作りやすいことを織り込むため。
  ``--null simple`` は λ = n_x · n_y / N（N は単位の数）。どちらで選んでも、もう一方での数を meta に残す
- 厳しい段：Bonferroni（全組で ``--alpha``、既定 1%）。緩い段：Benjamini–Hochberg（偽発見率 ``--fdr``、既定 5%）。
  どちらも、両方を含む単位が ``--min-k``（既定 3）本に満たない組は引かない（2 本だけなら同じ人の 2 回の投稿でも起きる）
- 地図上の距離：下位クラスタの重心（意見の x / y の平均）どうしの距離を、全意見の x / y の外接矩形の対角線で割ったもの。
  ``--far``（既定 0.1）以上を「離れた組」とする
- ``--embeddings``（任意）：下位クラスタの埋め込みの重心どうしの cosine と、全組の中での順位を足す。
  意味が近いのに地図で離れた組（散布図の裂け目）と、意味も離れた組を見分けるため。無ければ付けない

    uv run scripts/build_links_cooccurrence.py docs/data/team-mirai-2025/public_result.json \\
        --group-by pr_number \\
        --embeddings data/team-mirai-2025/kouchou-ai-embeddings-1536dim.parquet \\
        --out docs/links/team-mirai-2025.json
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from datetime import date
from pathlib import Path

import numpy as np
from scipy import sparse, stats


def group_key(argument: dict, group_by: str | None):
    if group_by is None:
        return argument.get("comment_id")
    return (argument.get("attributes") or {}).get(group_by)


def load_embeddings(path: Path, arg_ids: list[str]) -> np.ndarray:
    """(id, embedding) の parquet を意見の順に並べ、長さ 1 に正規化して返す（build_mandalart_tree.py と同じ読み方）。"""
    import pyarrow.parquet as pq

    table = pq.read_table(path, columns=["id", "embedding"])
    emb_ids = table["id"].to_pylist()
    flat = np.asarray(table["embedding"].combine_chunks().flatten(), dtype=np.float32)
    E = flat.reshape(len(emb_ids), -1)
    pos = {i: k for k, i in enumerate(emb_ids)}
    missing = [i for i in arg_ids if i not in pos]
    if missing:
        raise SystemExit(f"埋め込みが無い意見が {len(missing)} 件あります（例: {missing[:3]}）")
    X = E[[pos[i] for i in arg_ids]].astype(np.float64)
    X /= np.linalg.norm(X, axis=1, keepdims=True) + 1e-12
    return X


def benjamini_hochberg(p: np.ndarray, q: float) -> np.ndarray:
    order = np.argsort(p)
    m = len(p)
    ok = np.nonzero(p[order] <= q * np.arange(1, m + 1) / m)[0]
    if not len(ok):
        return np.zeros(m, dtype=bool)
    return p <= p[order][ok.max()]


def cooccurrence(rows: np.ndarray, cols: np.ndarray, n_rows: int, n_cols: int) -> np.ndarray:
    """単位×まとまりの 0/1 行列 B から B^T B（組ごとの、両方を含む単位の数）を作る。同じ組の重複は 1 に潰す。"""
    B = sparse.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(n_rows, n_cols))
    B.sum_duplicates()
    B.data[:] = 1
    return (B.T @ B).toarray()


def sig(v: float, digits: int = 3) -> float:
    if v == 0 or not math.isfinite(v):
        return float(v)
    return float(f"{v:.{digits}g}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("public_result", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--group-by", default=None, help="束ねる attributes の属性名。省略時は comment_id")
    parser.add_argument("--level", type=int, default=None, help="結ぶクラスタの段。省略時は最も下の段")
    parser.add_argument("--alpha", type=float, default=0.01, help="厳しい段：Bonferroni で全体の有意水準")
    parser.add_argument("--fdr", type=float, default=0.05, help="緩い段：Benjamini–Hochberg の偽発見率")
    parser.add_argument("--min-k", type=int, default=3, help="両方を含む単位がこの本数に満たない組は引かない")
    parser.add_argument("--far", type=float, default=0.1, help="「離れた組」とする地図上の距離（対角線に対する比）")
    parser.add_argument("--null", choices=["strat", "simple"], default="strat", help="偶然の見込み λ の出し方（上の説明）")
    parser.add_argument("--embeddings", type=Path, default=None, help="(id, embedding) の parquet。省略可")
    args = parser.parse_args()

    public = json.loads(args.public_result.read_text(encoding="utf-8"))
    if public.get("schema_version") != 1:
        print(f"warning: schema_version={public.get('schema_version')}（1 を想定）", file=sys.stderr)
    arguments = public["arguments"]
    clusters = public["clusters"]
    levels = sorted({c["level"] for c in clusters if c["level"] > 0})
    level = args.level or levels[-1]
    top_level = levels[0]

    # 下位クラスタと重心（意見の x / y の平均）
    xy = np.array([[a["x"], a["y"]] for a in arguments], dtype=np.float64)
    diag = float(math.hypot(*(xy.max(axis=0) - xy.min(axis=0))))
    leaf_of = [a["cluster_ids"][level] for a in arguments]
    top_of = [a["cluster_ids"][top_level] for a in arguments]
    leaf_ids = sorted(set(leaf_of), key=lambda c: (len(c), c))
    L = len(leaf_ids)
    li = {c: i for i, c in enumerate(leaf_ids)}
    lv = np.array([li[c] for c in leaf_of])
    size = np.bincount(lv, minlength=L)
    cen = np.zeros((L, 2))
    np.add.at(cen, lv, xy)
    cen /= size[:, None]
    parent = {}
    for a, c in zip(top_of, leaf_of):
        parent.setdefault(c, a)

    # 束ねる単位ごとの、下位クラスタの集合
    members: dict = defaultdict(lambda: defaultdict(list))  # 単位 -> まとまり -> 意見の添字
    skipped = 0
    for i, a in enumerate(arguments):
        g = group_key(a, args.group_by)
        if g is None:
            skipped += 1
            continue
        members[g][lv[i]].append(i)
    if not members:
        raise SystemExit(f"--group-by {args.group_by}：この属性を持つ意見が 1 件もない（arguments[].attributes の属性名を確かめる）")
    if skipped:
        print(f"warning: {args.group_by or 'comment_id'} の無い意見 {skipped} 件は外した", file=sys.stderr)
    keys = sorted(members, key=lambda g: (isinstance(g, str), g))
    N = len(keys)
    rows = np.concatenate([np.full(len(members[g]), r) for r, g in enumerate(keys)]).astype(np.int64)
    cols = np.concatenate([sorted(members[g]) for g in keys]).astype(np.int64)
    K = cooccurrence(rows, cols, N, L)
    n = np.diag(K).copy()  # 各まとまりを含む単位の数
    m = np.array([len(members[g]) for g in keys])  # 各単位が含むまとまりの数
    tops_per_group = np.array([len({top_of[i] for idx in members[g].values() for i in idx}) for g in keys])

    iu = np.triu_indices(L, 1)
    k = K[iu]
    T = len(k)
    lam_by = {"simple": n[iu[0]] * n[iu[1]] / N}
    B = sparse.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(N, L))
    LT = np.zeros((L, L))
    for s_ in np.unique(m):
        if s_ < 2:
            continue
        ns = np.asarray(B[np.nonzero(m == s_)[0]].sum(axis=0)).ravel()
        LT += np.outer(ns, ns) * (s_ - 1) / (s_ * int((m == s_).sum()))
    lam_by["strat"] = LT[iu]

    def select(lam: np.ndarray):
        # 補正は全 T 組で行い、そのあと本数の少ない組を落とす
        p_ = np.where(k > 0, stats.poisson.sf(k - 1, np.maximum(lam, 1e-12)), 1.0)
        enough = k >= args.min_k
        return p_, (p_ * T <= args.alpha) & enough, benjamini_hochberg(p_, args.fdr) & enough

    lam = lam_by[args.null]
    p, strict, loose = select(lam)
    keep = strict | loose
    other = "simple" if args.null == "strat" else "strat"
    _, strict_o, loose_o = select(lam_by[other])
    dist = np.hypot(*(cen[iu[0]] - cen[iu[1]]).T) / diag

    # 埋め込みでの近さ（任意）
    sim = sim_top = None
    space = None
    if args.embeddings is not None:
        X = load_embeddings(args.embeddings, [a["arg_id"] for a in arguments])
        C = np.zeros((L, X.shape[1]))
        np.add.at(C, lv, X)
        C /= np.linalg.norm(C, axis=1, keepdims=True) + 1e-12
        sim = (C @ C.T)[iu]
        # 全組のうち、この組と同じかより近い組の割合（小さいほど意味が近い）
        sim_top = 1.0 - (stats.rankdata(sim, method="min") - 1) / T
        space = f"embeddings_{X.shape[1]}d"

    # 見込みの出し方を変えたときの数（もう一方の λ で選び直す）。λ の合計と実際の共起数の合計の比も残す
    compare = {
        "null": other,
        "strict": int(strict_o.sum()),
        "strict_common": int((strict & strict_o).sum()),
        "loose": int((strict_o | loose_o).sum()),
        "lam_total_over_k_total": {name: sig(float(v.sum() / max(1, k.sum()))) for name, v in lam_by.items()},
    }

    groups_of = defaultdict(list)  # まとまり -> それを含む単位の番号（keys の順）
    for r, g in enumerate(keys):
        for c_ in members[g]:
            groups_of[c_].append(r)
    pairs = []
    for o in np.nonzero(keep)[0]:
        x, y = int(iu[0][o]), int(iu[1][o])
        ev = []
        for r in sorted(set(groups_of[x]) & set(groups_of[y])):
            mg = members[keys[r]]
            ev.append([keys[r], [arguments[i]["arg_id"] for i in mg[x]], [arguments[i]["arg_id"] for i in mg[y]]])
        pair = {
            "a": leaf_ids[x],
            "b": leaf_ids[y],
            "k": int(k[o]),
            "lam": sig(float(lam[o]), 4),
            "lift": sig(float(k[o] / lam[o])),
            "p": sig(float(p[o])),
            "d": round(float(dist[o]), 4),
            "tier": 1 if strict[o] else 2,
        }
        if sim is not None:
            pair["sim"] = round(float(sim[o]), 4)
            pair["sim_top"] = sig(float(sim_top[o]))
        pair["ev"] = ev
        pairs.append(pair)
    pairs.sort(key=lambda q: (q["tier"], q["p"]))

    # 上位クラスタの名前を置く位置（意見の x / y の平均）
    top_ids = sorted({c["id"] for c in clusters if c["level"] == top_level}, key=lambda c: (len(c), c))
    top_pos = defaultdict(list)
    for i, t in enumerate(top_of):
        top_pos[t].append(i)

    def pct(mask: np.ndarray) -> dict | None:
        if not mask.any():
            return None
        q = np.percentile(dist[mask], [50, 90, 99])
        return {"p50": round(float(q[0]), 4), "p90": round(float(q[1]), 4), "p99": round(float(q[2]), 4), "max": round(float(dist[mask].max()), 4)}

    same_top = np.array([parent[leaf_ids[a]] == parent[leaf_ids[b]] for a, b in zip(*iu)])
    far = dist >= args.far
    out = {
        "meta": {
            "n_arguments": len(arguments),
            "group_by": args.group_by or "comment_id",
            "group_source": "attributes" if args.group_by else "comment_id",
            "n_arguments_without_group": skipped,
            "level": level,
            "top_level": top_level,
            "n_groups": N,
            "n_groups_multi": int((m >= 2).sum()),
            "n_groups_multi_top": int((tops_per_group >= 2).sum()),
            "n_clusters": L,
            "n_tests": T,
            "n_pairs_observed": int((k > 0).sum()),
            "test": "poisson_upper_tail",
            "null": args.null,
            "alpha": args.alpha,
            "min_k": args.min_k,
            "fdr": args.fdr,
            "far": args.far,
            "diag": round(diag, 6),
            "embedding_space": space,
            "compare": compare,
            "counts": {
                "strict": int(strict.sum()),
                "loose": int(keep.sum()),
                "strict_far": int((strict & far).sum()),
                "loose_far": int((keep & far).sum()),
                "strict_cross_top": int((strict & ~same_top).sum()),
                "loose_cross_top": int((keep & ~same_top).sum()),
            },
            "distance": {"strict": pct(strict), "loose": pct(keep), "all_pairs": pct(np.ones(T, dtype=bool))},
            "built": date.today().isoformat(),
        },
        "clusters": [
            {"id": c, "x": round(float(cen[i, 0]), 4), "y": round(float(cen[i, 1]), 4), "n": int(n[i]), "size": int(size[i]), "top": parent[c]}
            for i, c in enumerate(leaf_ids)
        ],
        "top": [
            {"id": t, "x": round(float(xy[top_pos[t], 0].mean()), 4), "y": round(float(xy[top_pos[t], 1].mean()), 4), "size": len(top_pos[t])}
            for t in top_ids
            if t in top_pos
        ],
        "pairs": pairs,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")

    c = out["meta"]["counts"]
    print(f"単位={out['meta']['group_by']} N={N}（下位 2 つ以上 {out['meta']['n_groups_multi']}、上位 2 つ以上 {out['meta']['n_groups_multi_top']}）")
    print(f"組 {T}（k≥1 は {out['meta']['n_pairs_observed']}）：厳しい段 {c['strict']}（離れた組 {c['strict_far']}・上位をまたぐ {c['strict_cross_top']}）、緩い段まで {c['loose']}（離れた組 {c['loose_far']}・上位をまたぐ {c['loose_cross_top']}）")
    print(f"距離（対角線比）厳しい段 {out['meta']['distance']['strict']}・緩い段 {out['meta']['distance']['loose']}・全組 {out['meta']['distance']['all_pairs']}")
    print(f"見込み {args.null}。λ の合計 ÷ 共起数の合計：{compare['lam_total_over_k_total']}。{other} で選ぶと厳しい段 {compare['strict']}（共通 {compare['strict_common']}）、緩い段まで {compare['loose']}")
    if sim_top is not None:
        st = np.array([q["sim_top"] for q in pairs if q["tier"] == 1])
        sf = np.array([q["sim_top"] for q in pairs if q["tier"] == 1 and q["d"] >= args.far])
        print(f"埋め込みの近さ（全組を近い順に並べて上から何割目か）：厳しい段の中央値 {np.median(st):.4f}、うち離れた組 {np.round(np.sort(sf), 3).tolist()}")
    size_kb = args.out.stat().st_size / 1024
    print(f"wrote {args.out} ({size_kb:.0f} KB): pairs={len(pairs)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
