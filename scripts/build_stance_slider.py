"""賛否スライダー用の事前計算：賛否スコアに重みを掛けて埋め込みに 1 次元足し、重みごとに UMAP の 2 次元配置を作る。

元は tokoroten の「センチメント次元の付与（sentiment-weighted UMAP）」
（https://sentiment-weighted-umap-animation.netlify.app/ 、posinega_separation.py）。
- 賛否スコア s（-1〜+1）に重み w を掛け、埋め込みの末尾に足す：[埋め込み…, w·s]。w を上げるほど同じ話題の賛成と反対が離れる
- 重み 0 で UMAP を学習し、次の重みは前の重みの座標を初期値に少ないエポックで学習し直す（元と同じ「連続性を保つトリック」）。
  ただし元の設定（学習率 1.0）のままだと、重みを変えずに学習し直しただけで点が配置の幅の 12%（中央値）も漂った。
  学習率を 0.1 に下げ、各段を前の段に縮尺・回転・平行移動だけで重ねる（Procrustes）と、漂いは 2.5% になり、
  重み 1.5 での賛否の分かれ方（下の sep）はほぼ変わらなかった（移民のまとまりで 0.77 → 0.72）
- 表示側は重みの段のあいだを線形補間するだけ

入力は広聴AI の public_result.json。賛否スコアは本体の任意ステップ opinion_stance と同じ arguments[].attributes.stance_score を読む
（判定不能で属性の無い意見は s = 0 として扱う）。--embeddings に (id, embedding) の parquet を渡すとその埋め込みを、
省略すると public_result.json の 2 次元配置 x/y を土台にする。

重みごとに 2 つの数値も残す（本体 docs の「賛否で分かれたか」と「トピックの近さが壊れていないか」を同時に見る、に当たる）
- sep：賛成（s > 0.2）と反対（s < -0.2）を 2 群としたときの 2 次元でのシルエット係数。上がるほど賛否で分かれている
- keep：土台（重み 0 の埋め込み）での近傍 15 件のうち、2 次元配置でも近傍 15 件に入っている割合。下がるほど話題の近さが壊れている

    uv run --group umap scripts/build_stance_slider.py docs/data/team-mirai-2025-stance/1_13/public_result.json \\
        --embeddings data/team-mirai-2025/kouchou-ai-embeddings-1536dim.parquet \\
        --out docs/slider/team-mirai-2025-1_13.json
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from datetime import date
from pathlib import Path

import numpy as np

K_NEIGHBORS = 15


def load_base(public: dict, embeddings_path: Path | None) -> tuple[np.ndarray, str, str]:
    ids = [a["arg_id"] for a in public["arguments"]]
    if embeddings_path is None:
        X = np.array([[a["x"], a["y"]] for a in public["arguments"]], dtype=np.float64)
        X = (X - X.mean(0)) / (X.std(0) + 1e-12)
        return X, "2d_layout", "euclidean"
    import pyarrow.parquet as pq

    table = pq.read_table(embeddings_path, columns=["id", "embedding"])
    emb_ids = table["id"].to_pylist()
    pos = {i: k for k, i in enumerate(emb_ids)}
    missing = [i for i in ids if i not in pos]
    if missing:
        raise SystemExit(f"埋め込みが無い意見が {len(missing)} 件あります（例: {missing[:3]}）")
    flat = np.asarray(table["embedding"].combine_chunks().flatten(), dtype=np.float32)
    E = flat.reshape(len(emb_ids), -1)[[pos[i] for i in ids]].astype(np.float64)
    E /= np.linalg.norm(E, axis=1, keepdims=True) + 1e-12
    return E, f"embeddings_{E.shape[1]}d", "cosine"


def align(Y: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """縮尺・回転・平行移動だけで Y を ref に重ねる（Procrustes）。UMAP が初期値の座標を正規化し直す分の見かけの動きを消す。"""
    A, B = Y - Y.mean(0), ref - ref.mean(0)
    U, S, Vt = np.linalg.svd(A.T @ B)
    return (A @ (U @ Vt)) * (S.sum() / (A**2).sum()) + ref.mean(0)


def neighbors(X: np.ndarray, metric: str, k: int) -> np.ndarray:
    from sklearn.neighbors import NearestNeighbors

    nn = NearestNeighbors(n_neighbors=k + 1, metric=metric).fit(X)
    return nn.kneighbors(X, return_distance=False)[:, 1:]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("public_result", type=Path)
    ap.add_argument("--embeddings", type=Path, default=None)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--max-weight", type=float, default=1.5)
    ap.add_argument("--step", type=float, default=0.05, help="重みの刻み（元の実装と同じ 0.05）")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--learning-rate", type=float, default=0.1, help="2 段目以降の学習率（元の実装は 1.0）")
    args = ap.parse_args()

    import umap
    from sklearn.metrics import silhouette_score

    warnings.filterwarnings("ignore", category=UserWarning)
    public = json.loads(args.public_result.read_text())
    s = np.array([float((a.get("attributes") or {}).get("stance_score", 0.0)) for a in public["arguments"]])
    if not any("stance_score" in (a.get("attributes") or {}) for a in public["arguments"]):
        sys.exit("arguments[].attributes.stance_score が 1 件も無い（opinion_stance を有効にしたレポートか、add_opinion_stance.py の出力を渡す）")
    X, space, metric = load_base(public, args.embeddings)
    n_steps = int(round(args.max_weight / args.step))
    weights = [round(i * args.step, 4) for i in range(n_steps + 1)]
    base_nb = neighbors(X, metric, K_NEIGHBORS)
    side = np.where(s > 0.2, 1, np.where(s < -0.2, -1, 0))  # 本体の opinion_stance の集計と同じく ±0.2 までは中立
    polar = side != 0

    frames, prev = [], None
    for w in weights:
        Z = np.hstack([X, (w * s)[:, None]])
        if prev is None:
            reducer = umap.UMAP(n_components=2, n_neighbors=15, min_dist=0.1, metric=metric, random_state=args.seed)
        else:  # 前の重みの座標を初期値に、少ないエポック・低い学習率で学習し直す
            reducer = umap.UMAP(n_components=2, n_neighbors=15, min_dist=0.1, metric=metric, random_state=args.seed,
                                init=prev, n_epochs=50, learning_rate=args.learning_rate)
        Y = reducer.fit_transform(Z)
        Y = Y - Y.mean(0) if prev is None else align(Y, prev)
        prev = Y
        nb2 = neighbors(Y, "euclidean", K_NEIGHBORS)
        keep = float(np.mean([len(set(a) & set(b)) / K_NEIGHBORS for a, b in zip(base_nb, nb2)]))
        sep = float(silhouette_score(Y[polar], side[polar])) if len(set(side[polar])) == 2 else None
        frames.append({"w": w, "x": [round(float(v), 2) for v in Y[:, 0]], "y": [round(float(v), 2) for v in Y[:, 1]],
                       "sep": None if sep is None else round(sep, 3), "keep": round(keep, 3)})
        print(f"  w={w:.2f} sep={sep if sep is None else round(sep, 3)} keep={keep:.3f}", file=sys.stderr)

    out = {
        "meta": {
            "method": "sentiment-weighted UMAP（tokoroten）。埋め込みに w·stance_score を 1 次元足し、前の重みの座標を初期値に学習し直して前の段に重ねる",
            "source": "https://sentiment-weighted-umap-animation.netlify.app/",
            "space": space, "metric": metric, "seed": args.seed, "n_neighbors": 15, "min_dist": 0.1,
            "refit": {"n_epochs": 50, "learning_rate": args.learning_rate, "align": "procrustes"},
            "n_arguments": len(s), "n_scored": int(sum("stance_score" in (a.get("attributes") or {}) for a in public["arguments"])),
            "n_for": int((side == 1).sum()), "n_against": int((side == -1).sum()),
            "target": (public.get("config") or {}).get("question"), "built": date.today().isoformat(),
        },
        "arg_ids": [a["arg_id"] for a in public["arguments"]],
        "frames": frames,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, ensure_ascii=False, separators=(",", ":")))
    print(f"wrote {args.out} ({args.out.stat().st_size / 1e6:.2f} MB)", file=sys.stderr)


if __name__ == "__main__":
    main()
