"""付箋ズーム用の配置を作る：各意見の付箋の位置と「どの拡大率から付箋として出すか」を事前に決める。

考え方（地図アプリの地名ラベルと同じ）
- 付箋は画面上で一定の大きさ（拡大しても文字が読める）。置く位置は世界座標で固定する
- 引いた状態では全部は置けないので、優先度の高い意見から「重ならないなら出す」を決める
- この判定を拡大率の段ごとに、引いた側から順に行う。一度出た付箋は、それより寄った段でも必ず出る
  （寄るほど付箋どうしの画面上の距離は広がるので、重なりは増えない）
- その結果を意見ごとの「出始める拡大率」として保存する。表示側は拡大率と比べるだけなので、
  拡大・移動しても付箋が入れ替わったり跳んだりしない

最終段（最も寄った状態）で全意見が重ならずに並ぶよう、位置はオリジナル（canvas_kozaneba_prototype）と
同じく格子に寄せ、同じマスが埋まっていたら螺旋状に一番近い空きマスへずらす。

優先度
1. 上位クラスタの代表（各クラスタの中心に最も近い意見）。付箋に上位クラスタの表札をつける
2. 下位クラスタの代表。付箋に下位クラスタの表札をつける
3. 残り。下位クラスタごとに中心に近い順の順位をつけ、順位 0 → 1 → 2 … と全クラスタを順に回す
   （引いた状態でも、特定のクラスタだけが付箋で埋まらないように）

    uv run scripts/build_sticky_layout.py docs/data/team-mirai-2025/public_result.json \
        --out docs/sticky/team-mirai-2025.layout.json
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from datetime import date
from pathlib import Path

import numpy as np

NOTE_W, NOTE_H, GAP = 140, 76, 6  # 画面上の付箋の大きさ（px）と隙間
STEPS_PER_OCTAVE = 8  # 拡大率の段の細かさ（2 倍ごとに 8 段）


def priorities(public: dict) -> tuple[list[int], dict[int, int]]:
    """意見の添字を優先度順に並べる。あわせて表札をつける意見 {添字: 1 or 2} を返す。"""
    args = public["arguments"]
    xy = np.array([[a["x"], a["y"]] for a in args])
    levels = sorted({c["level"] for c in public["clusters"] if c["level"] > 0})
    l1, leaf = levels[0], levels[-1]
    size = {c["id"]: c["value"] for c in public["clusters"]}

    def members(level: int) -> dict[str, list[int]]:
        out: dict[str, list[int]] = defaultdict(list)
        for i, a in enumerate(args):
            out[a["cluster_ids"][level]].append(i)
        return out

    def ranked(idx: list[int]) -> list[int]:
        pts = xy[idx]
        d = ((pts - pts.mean(axis=0)) ** 2).sum(axis=1)
        return [idx[k] for k in np.argsort(d, kind="stable")]

    order: list[int] = []
    head: dict[int, int] = {}
    seen: set[int] = set()
    for tier, level in ((1, l1), (2, leaf)):
        groups = members(level)
        for cid in sorted(groups, key=lambda c: -size.get(c, len(groups[c]))):
            for i in ranked(groups[cid]):
                if i not in seen:
                    order.append(i)
                    seen.add(i)
                    head[i] = tier
                    break
    leaf_groups = members(leaf)
    queues = [ranked(leaf_groups[c]) for c in sorted(leaf_groups, key=lambda c: -size.get(c, 0))]
    rank = 0
    while any(rank < len(q) for q in queues):
        for q in queues:
            if rank < len(q) and q[rank] not in seen:
                order.append(q[rank])
                seen.add(q[rank])
        rank += 1
    return order, head


def snap(xy: np.ndarray, order: list[int], cw: float, ch: float) -> tuple[np.ndarray, float]:
    """優先度順に格子へ寄せる。埋まっていたら螺旋状に一番近い空きマスへ。"""
    taken: set[tuple[int, int]] = set()
    out = np.zeros_like(xy)
    moved = []
    for i in order:
        gx0, gy0 = int(np.floor(xy[i, 0] / cw)), int(np.floor(xy[i, 1] / ch))
        r = 0
        while True:
            best = None
            for dx in range(-r, r + 1):
                for dy in range(-r, r + 1):
                    if max(abs(dx), abs(dy)) != r or (gx0 + dx, gy0 + dy) in taken:
                        continue
                    cand = ((gx0 + dx + 0.5) * cw, (gy0 + dy + 0.5) * ch)
                    dist = (cand[0] - xy[i, 0]) ** 2 + (cand[1] - xy[i, 1]) ** 2
                    if best is None or dist < best[0]:
                        best = (dist, gx0 + dx, gy0 + dy, cand)
            if best:
                taken.add((best[1], best[2]))
                out[i] = best[3]
                moved.append(r)
                break
            r += 1
    return out, float(np.mean(moved))


def min_zooms(pos: np.ndarray, order: list[int], z_min: float, z_full: float) -> np.ndarray:
    """拡大率の段ごとに、引いた側から「重ならなければ出す」を優先度順に決める。"""
    n = len(pos)
    zmin = np.full(n, np.inf)
    steps = int(np.ceil(np.log2(z_full / z_min) * STEPS_PER_OCTAVE))
    zs = [z_min * 2 ** (s / STEPS_PER_OCTAVE) for s in range(steps)] + [z_full]
    w, h = NOTE_W + GAP, NOTE_H + GAP
    for z in zs:
        cw, ch = w / z, h / z  # この拡大率で付箋 1 枚が占める世界座標の幅・高さ
        grid: dict[tuple[int, int], list[int]] = defaultdict(list)

        def fits(i: int) -> bool:
            gx, gy = int(np.floor(pos[i, 0] / cw)), int(np.floor(pos[i, 1] / ch))
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    for j in grid.get((gx + dx, gy + dy), ()):
                        if abs(pos[i, 0] - pos[j, 0]) < cw - 1e-9 and abs(pos[i, 1] - pos[j, 1]) < ch - 1e-9:
                            return False
            return True

        def put(i: int) -> None:
            grid[(int(np.floor(pos[i, 0] / cw)), int(np.floor(pos[i, 1] / ch)))].append(i)

        for i in order:  # 既に出ている付箋は無条件で残す（寄るほど重ならなくなる）
            if np.isfinite(zmin[i]):
                put(i)
        for i in order:
            if not np.isfinite(zmin[i]) and fits(i):
                zmin[i] = z
                put(i)
        if np.isfinite(zmin).all():
            break
    zmin[~np.isfinite(zmin)] = z_full
    return zmin


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("public_result", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--full-zoom-ratio",
        type=float,
        default=60.0,
        help="全意見が付箋になる拡大率を「全体が 1000px に収まる拡大率」の何倍にするか（大きいほど位置のずれが小さい）",
    )
    args = parser.parse_args()

    public = json.loads(args.public_result.read_text(encoding="utf-8"))
    xy = np.array([[a["x"], a["y"]] for a in public["arguments"]], dtype=np.float64)
    span = float(max(np.ptp(xy[:, 0]), np.ptp(xy[:, 1])))
    z_ref = 1000.0 / span  # 全体が 1000px に収まる拡大率（px / 座標の単位）
    z_full = z_ref * args.full_zoom_ratio
    order, head = priorities(public)
    pos, mean_moved = snap(xy, order, (NOTE_W + GAP) / z_full, (NOTE_H + GAP) / z_full)
    zmin = min_zooms(pos, order, z_min=z_ref / 4, z_full=z_full)

    rank = np.empty(len(order), dtype=int)
    rank[order] = np.arange(len(order))
    out = {
        "meta": {
            "note": {"w": NOTE_W, "h": NOTE_H},
            "z_ref": z_ref,
            "z_full": z_full,
            "steps_per_octave": STEPS_PER_OCTAVE,
            "mean_spiral_radius": round(mean_moved, 3),
            "n_arguments": len(order),
            "built": date.today().isoformat(),
        },
        "x": [round(float(v), 5) for v in pos[:, 0]],
        "y": [round(float(v), 5) for v in pos[:, 1]],
        "z": [round(float(v), 4) for v in zmin],
        "rank": rank.tolist(),
        "head": [head.get(i, 0) for i in range(len(order))],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, separators=(",", ":")), encoding="utf-8")
    ratios = zmin / z_ref
    for r in (1, 2, 4, 8, 16, 32):
        print(f"  拡大率 {r:>2} 倍（1000px 基準）で付箋になる意見: {(ratios <= r).sum():>6}")
    print(f"wrote {args.out}: z_full={z_full:.0f} mean_spiral_radius={mean_moved:.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
