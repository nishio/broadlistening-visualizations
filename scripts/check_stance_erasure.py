"""賛否スライダーの「逆」：賛否を無視した配置は、元の広聴AI の配置とどう違うかを確かめる。

賛否スライダーの重み 0 は、埋め込みをそのまま UMAP にかけた配置で、元の広聴AI と同じ考え方になる。
重みを負にしても、足すのは 1 次元だけなので距離は |w| で決まり、正と同じだけ分かれる（無視にはならない）。
本当に無視するには、埋め込みから賛否と相関する方向を取り除く（線形の概念消去）。ここではそれを試して、次を出す。

- 埋め込みから賛否（賛成 > 0.2 / 反対 < -0.2）を当てられるか（AUC）。取り除く方向を決める意見・当てる分類器を学習する意見・
  評価する意見を分けて測る（同じ意見で決めて同じ意見で測ると、逆向きに当てられる状態を作るだけで AUC が 0 近くに見える）
  - meandiff：賛成と反対の平均の差の方向を k 回取り除く
  - logreg：ロジスティック回帰の重みの方向を k 回取り除く（INLP の簡易版）
- 消去した埋め込みで UMAP をかけ直した配置と、元の埋め込みの配置（重み 0）とで
  - 賛否の分かれ（2 次元でのシルエット係数）と、話題の近さの残り（元の埋め込みで近い 15 件が 2 次元でも近い割合）
  - 2 つの配置で近い 15 件がどれだけ重なるか。目安として、元の埋め込みのまま seed だけ変えた配置との重なりも出す

    uv run --group umap scripts/check_stance_erasure.py \\
        --embeddings data/team-mirai-2025/kouchou-ai-embeddings-1536dim.parquet \\
        docs/data/team-mirai-2025-stance/{1_13,1_1,1_20,1_19}/public_result.json
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_stance_slider import K_NEIGHBORS, load_base, neighbors  # noqa: E402


def directions(X: np.ndarray, y: np.ndarray, method: str, k: int) -> list[np.ndarray]:
    from sklearn.linear_model import LogisticRegression

    V, Xc = [], X.copy()
    for _ in range(k):
        if method == "meandiff":
            v = Xc[y].mean(0) - Xc[~y].mean(0)
        else:
            v = LogisticRegression(C=1.0, max_iter=2000).fit(Xc, y).coef_[0]
        v = v / (np.linalg.norm(v) + 1e-12)
        V.append(v)
        Xc = Xc - np.outer(Xc @ v, v)
    return V


def project(X: np.ndarray, V: list[np.ndarray]) -> np.ndarray:
    for v in V:
        X = X - np.outer(X @ v, v)
    return X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)


def heldout_auc(X: np.ndarray, y: np.ndarray, method: str, k: int, folds: int = 5) -> float:
    """5 分割の学習側をさらに層化して 2 つに割り、片方で取り除く方向、もう片方で分類器を決め、評価側で AUC を測る。"""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedKFold, train_test_split

    out = []
    for tr, te in StratifiedKFold(folds, shuffle=True, random_state=0).split(X, y):
        a, b = train_test_split(tr, test_size=0.5, stratify=y[tr], random_state=0)
        V = directions(X[a], y[a], method, k) if k else []
        clf = LogisticRegression(C=1.0, max_iter=2000, class_weight="balanced").fit(project(X[b], V), y[b])
        out.append(roc_auc_score(y[te], clf.decision_function(project(X[te], V))))
    return float(np.mean(out))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("public_results", type=Path, nargs="+")
    ap.add_argument("--embeddings", type=Path, required=True)
    ap.add_argument("--layout-method", default="logreg", choices=["meandiff", "logreg"])
    ap.add_argument("--layout-k", type=int, default=10, help="配置を作るときに取り除く方向の数")
    args = ap.parse_args()

    import umap
    from sklearn.metrics import silhouette_score

    warnings.filterwarnings("ignore")
    fit = lambda X, seed: umap.UMAP(n_neighbors=15, min_dist=0.1, metric="cosine", random_state=seed).fit_transform(X)  # noqa: E731
    overlap = lambda A, B: float(np.mean([len(set(a) & set(b)) / K_NEIGHBORS for a, b in zip(A, B)]))  # noqa: E731

    for path in args.public_results:
        P = json.loads(path.read_text())
        X, _, metric = load_base(P, args.embeddings)
        s = np.array([float((a.get("attributes") or {}).get("stance_score", 0.0)) for a in P["arguments"]])
        side = np.where(s > 0.2, 1, np.where(s < -0.2, -1, 0))
        pol = side != 0
        y = side[pol] > 0
        name = path.parent.name
        aucs = {"元": heldout_auc(X[pol], y, "meandiff", 0)}
        for m in ["meandiff", "logreg"]:
            for k in [1, 3, 10]:
                aucs[f"{m}×{k}"] = heldout_auc(X[pol], y, m, k)
        print(f"[{name}] 賛成 {int(y.sum())}・反対 {int((~y).sum())}  AUC " + " | ".join(f"{k} {v:.2f}" for k, v in aucs.items()))

        # 配置は、賛否のついた全意見で取り除く方向を決めて、全意見に当てる
        Xe = project(X, directions(X[pol], y, args.layout_method, args.layout_k))
        base_nb = neighbors(X, metric, K_NEIGHBORS)
        Y0, Y0b, Y1 = fit(X, 42), fit(X, 7), fit(Xe, 42)
        n0, n0b, n1 = (neighbors(Y, "euclidean", K_NEIGHBORS) for Y in (Y0, Y0b, Y1))
        sep = lambda Y: float(silhouette_score(Y[pol], side[pol]))  # noqa: E731
        print(f"[{name}] 配置（{args.layout_method}×{args.layout_k} を取り除く）"
              f"賛否の分かれ {sep(Y0):.2f} → {sep(Y1):.2f}・話題の近さの残り {overlap(base_nb, n0):.2f} → {overlap(base_nb, n1):.2f}・"
              f"元の配置との近傍の重なり {overlap(n0, n1):.2f}（seed 違い {overlap(n0, n0b):.2f}）")


if __name__ == "__main__":
    main()
