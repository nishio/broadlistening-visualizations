"""自分の声を探す（docs/find/）の確かめ：ページと同じ字面の近さを Python でやり直し、近さの目安の根拠を出す。

ページは事前計算を使わない（ブラウザが public_result.json から索引を作る）。このスクリプトはページの外で、
ページの JavaScript と同じ文字の組・同じ TF-IDF の式を使って次を Markdown で出す。LLM は使わない。

1. 文字の組の選び方の比べ：意見を問いにして、同じ投稿の意見を除いた近い 10 件が、広聴AI の同じ下位・上位の
   まとまりに入る割合（--embeddings があれば、埋め込みの近い 10 件との重なりも）
2. 近さ別に、2 件が同じまとまりに入る割合（「近い」「とても近い」の目安の根拠）
3. 例文と、データに無い話題の文での結果
4. 地図の印の位置：近い意見の重み付き重心と、重み付き幾何中央値

    uv run scripts/build_find_check.py docs/data/team-mirai-2025/public_result.json --group-by pr_number \\
        --embeddings data/team-mirai-2025/kouchou-ai-embeddings-1536dim.parquet

--group-by は「同じ投稿」とみなして近い 10 件から除く属性（トップレベルの項目か attributes の名前。省略時は comment_id）。
--dump <json> で、例文の近い 10 件（arg_id と近さ）を書き出す（ブラウザの結果と突き合わせる用）。
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
import time
import unicodedata
from collections import Counter
from pathlib import Path

import numpy as np
import scipy.sparse as sp

# ---- ページ（docs/find/index.html）と同じ文字の組 ----
# 近さの目安（少し重なる・近い・とても近い）。ページの LEVELS と同じ値
LEVELS = {"ja": (0.20, 0.30, 0.45), "en": (0.30, 0.40, 0.55)}
MARK_K = 20
STOP_EN = set(
    (
        "a an the of to in on for and or nor but is are was were be been being am it its this that these those there here with by as at from not no "
        "they them their we us our i me my you your he him his she her which who whom whose what when where why how than then so such "
        "can could should would will may might must shall do does did have has had more most very also into about over under between among "
        "within without through during before after above below up down out off again further once if because while until against "
        "all any both each few other some own same only too just please want wants wish hope think feel"
    ).split()
)
JA_TAIL = re.compile(
    r"(?:して|て|で)?(?:ほしい|欲しい|ください|下さい|もらいたい|いただきたい|頂きたい)(?:です|と思います|と思う|のです)?(?=[、。,.!?\s]|$)"
    r"|(?:と思います|と思う|と考えます|です|ます|でした|ました)(?=[、。,.!?\s]|$)"
)


def norm(s: str) -> str:
    return unicodedata.normalize("NFKC", s).lower()


def segments(s: str) -> list[str]:
    """JavaScript の /[\\p{L}\\p{N}]+/gu と同じ：文字と数字の続き"""
    out, cur = [], []
    for ch in norm(s):
        if unicodedata.category(ch)[0] in "LN":
            cur.append(ch)
        elif cur:
            out.append("".join(cur))
            cur = []
    if cur:
        out.append("".join(cur))
    return out


def is_hira(ch: str) -> bool:
    return "぀" <= ch <= "ゟ"


def grams_ja(s: str) -> list[str]:
    """ページの日本語：2 文字の組と、ひらがな以外の 1 文字"""
    g = []
    for seg in segments(s):
        for i, ch in enumerate(seg):
            if not is_hira(ch):
                g.append(ch)
            if i + 1 < len(seg):
                g.append(seg[i : i + 2])
    return g


def grams_en(s: str) -> list[str]:
    """ページの英語：よく使う語を除いた単語と、その単語の前後に空白を付けた 3 文字の組"""
    g = []
    for w in segments(s):
        if w in STOP_EN:
            continue
        g.append("w:" + w)
        p = " " + w + " "
        g.extend(p[i : i + 3] for i in range(len(p) - 2))
    return g


def prep_ja(q: str) -> str:
    return JA_TAIL.sub(" ", norm(q))


# ---- 比べる候補 ----
def _chars(ns):
    def f(s):
        g = []
        for seg in segments(s):
            for n in ns:
                if n == 1:
                    g.extend(ch for ch in seg)
                else:
                    g.extend(seg[i : i + n] for i in range(len(seg) - n + 1))
        return g

    return f


def _en_words(s):
    return ["w:" + w for w in segments(s) if w not in STOP_EN]


def _en_c3_all(s):
    g = []
    for w in segments(s):
        p = " " + w + " "
        g.extend(p[i : i + 3] for i in range(len(p) - 2))
    return g


CANDIDATES = {
    "ja": [
        ("2 文字", _chars((2,))),
        ("2・3 文字", _chars((2, 3))),
        ("3 文字", _chars((3,))),
        ("1・2 文字（1 文字はすべて）", _chars((1, 2))),
        ("1・2 文字（1 文字はひらがな以外）＝ページ", grams_ja),
    ],
    "en": [
        ("単語（よく使う語を除く）", _en_words),
        ("3 文字（すべての単語）", _en_c3_all),
        ("単語＋3 文字（よく使う語を除く）＝ページ", grams_en),
    ],
}


class Index:
    """ページの buildIndex と同じ：重み (1 + ln tf)·ln(N/df) を長さ 1 にする。問いの未知の組は df = 1 とみなして長さに入れる"""

    def __init__(self, texts: list[str], grams):
        self.grams = grams
        vocab: dict[str, int] = {}
        rows, cols, vals = [], [], []
        for d, text in enumerate(texts):
            for g, c in Counter(grams(text)).items():
                rows.append(d)
                cols.append(vocab.setdefault(g, len(vocab)))
                vals.append(c)
        n = len(texts)
        M = sp.csr_matrix((np.array(vals, dtype=np.float64), (rows, cols)), shape=(n, len(vocab)))
        df = np.bincount(M.indices, minlength=len(vocab))
        self.n, self.vocab, self.nnz = n, vocab, M.nnz
        self.idf = np.log(n / df)
        self.idf_oov = math.log(n)
        M.data = 1 + np.log(M.data)
        M = sp.csr_matrix(M.multiply(self.idf))
        norms = np.sqrt(np.asarray(M.multiply(M).sum(axis=1)).ravel())
        norms[norms == 0] = 1
        self.M = sp.csr_matrix(sp.diags(1 / norms) @ M)

    def scores(self, text: str) -> np.ndarray:
        idx, val, n2 = [], [], 0.0
        for g, c in Counter(self.grams(text)).items():
            j = self.vocab.get(g)
            x = (1 + math.log(c)) * (self.idf_oov if j is None else self.idf[j])
            n2 += x * x
            if j is not None and x > 0:
                idx.append(j)
                val.append(x)
        if not idx or n2 == 0:
            return np.zeros(self.n)
        q = np.zeros(len(self.vocab))
        q[idx] = np.array(val) / math.sqrt(n2)
        return self.M @ q


# ---- 読み込み ----
def group_of(a: dict, key: str):
    v = a[key] if key in a else (a.get("attributes") or {}).get(key)
    return None if v is None else str(v)


def load_embeddings(path: Path, arguments: list[dict]) -> np.ndarray:
    import pyarrow.parquet as pq

    t = pq.read_table(path, columns=["id", "embedding"])
    ids = t["id"].to_pylist()
    E = np.asarray(t["embedding"].combine_chunks().flatten(), dtype=np.float32).reshape(len(ids), -1)
    pos = {i: k for k, i in enumerate(ids)}
    missing = [a["arg_id"] for a in arguments if a["arg_id"] not in pos]
    if missing:
        raise SystemExit(f"埋め込みが無い意見が {len(missing)} 件あります（例: {missing[:3]}）")
    E = E[[pos[a["arg_id"]] for a in arguments]]
    E /= np.linalg.norm(E, axis=1, keepdims=True) + 1e-12
    return E


def topk_excluding(s: np.ndarray, same: np.ndarray, k: int) -> np.ndarray:
    s = s.copy()
    s[same] = -np.inf
    top = np.argpartition(-s, k)[:k]
    return top[np.argsort(-s[top], kind="stable")]


def log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


# ---- 例文 ----
QUERIES_JA = [
    "夫婦別姓を早く導入してほしい", "原発の再稼働に反対です", "保育士の給料を上げてほしい", "マイナンバーカードが使いにくい", "AI で行政を効率化してほしい",
    "消費税を下げてほしい", "子育て支援をもっと充実させてほしい", "奨学金の返済が苦しい", "ガソリン税が高すぎる", "介護職員が足りない",
    "いじめをなくしてほしい", "外国人の受け入れを増やすべき", "選挙でネット投票ができるようにしてほしい", "教員の長時間労働をなんとかして",
    "地方の公共交通がなくなって困っている", "同性婚を認めてほしい", "防衛費を増やすべきではない", "年金だけでは生活できない",
    "がん検診を無料にしてほしい", "農家の後継者がいない",
]
OFF_JA = ["ゾンビ対策をしてほしい", "猫がかわいい", "今日の晩ごはんはカレーです", "宇宙人と交信したい", "推しのライブに行きたい", "富士山に登りたい", "ピアノを習いたい", "宝くじが当たりますように"]
QUERIES_EN = [
    "I want selective separate surnames for married couples introduced soon", "I oppose restarting nuclear power plants",
    "Please raise the salaries of nursery teachers", "The My Number Card is hard to use", "Use AI to make government more efficient",
    "Lower the consumption tax", "Student loan repayment is a heavy burden", "There are not enough care workers", "Allow same-sex marriage", "Farmers have no successors",
]
OFF_EN = ["Zombies are scary", "My cat is cute", "I had curry for dinner", "I want to climb Mount Fuji", "I want to learn the piano"]


def geometric_median(P: np.ndarray, w: np.ndarray, iters: int = 50) -> np.ndarray:
    c = (P * w[:, None]).sum(0) / w.sum()
    for _ in range(iters):
        d = np.maximum(np.linalg.norm(P - c, axis=1), 1e-9)
        q = w / d
        c = (P * q[:, None]).sum(0) / q.sum()
    return c


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("public_result", type=Path)
    ap.add_argument("--group-by", default="comment_id", help="同じ投稿とみなして除く属性（省略時は comment_id）")
    ap.add_argument("--embeddings", type=Path, help="(id, embedding) の parquet。埋め込みの近い 10 件との重なりを出す")
    ap.add_argument("--sample", type=int, default=2000, help="問いにする意見の数（乱数の種は 0）")
    ap.add_argument("--dump", type=Path, help="例文の近い 10 件を JSON で書き出す")
    args = ap.parse_args()

    P = json.loads(args.public_result.read_text(encoding="utf-8"))
    A = P["arguments"]
    n = len(A)
    max_level = max(c["level"] for c in P["clusters"])
    clusters = {c["id"]: c for c in P["clusters"]}
    lo = np.array([a["cluster_ids"][max_level] for a in A])
    up = np.array([a["cluster_ids"][1] for a in A])
    grp = np.array([group_of(a, args.group_by) or f"__{k}" for k, a in enumerate(A)])
    en_args = ((P.get("translations") or {}).get("en") or {}).get("arguments")
    texts = {"ja": [a["argument"] for a in A]}
    if en_args:
        texts["en"] = [en_args.get(a["arg_id"], a["argument"]) for a in A]

    rng = np.random.default_rng(0)
    S = rng.choice(n, min(args.sample, n), replace=False)
    same = {i: np.flatnonzero(grp == grp[i]) for i in S}
    K = 10

    emb_nb = None
    if args.embeddings:
        log("埋め込みを読む")
        E = load_embeddings(args.embeddings, A)
        emb_nb = {}
        for ch in range(0, len(S), 500):
            Q = E[S[ch : ch + 500]] @ E.T
            for r, i in enumerate(S[ch : ch + 500]):
                emb_nb[i] = topk_excluding(Q[r], same[i], K)
        del E

    print(f"# 自分の声を探す：確かめ（{args.public_result}・意見 {n:,} 件・問いにした意見 {len(S):,} 件・同じ {args.group_by} の意見は除く）\n")

    # 1. 文字の組の選び方
    print("## 1. 文字の組の選び方\n")
    print("| 言語 | 文字の組 | 種類 | 延べ | 近い 10 件が同じ下位のまとまり | 同じ上位のまとまり | 埋め込みの近い 10 件との重なり |")
    print("|---|---|---|---|---|---|---|")
    if emb_nb is not None:
        e_lo = np.mean([np.mean(lo[emb_nb[i]] == lo[i]) for i in S])
        e_up = np.mean([np.mean(up[emb_nb[i]] == up[i]) for i in S])
        print(f"| 参考 | 埋め込み（1536 次元の cosine） | | | {e_lo:.3f} | {e_up:.3f} | 1 |")
    chosen = {}
    for lang, cands in CANDIDATES.items():
        if lang not in texts:
            continue
        for name, f in cands:
            t0 = time.time()
            ix = Index(texts[lang], f)
            s_lo, s_up, ov = [], [], []
            for ch in range(0, len(S), 500):
                sc = (ix.M[S[ch : ch + 500]] @ ix.M.T).toarray()
                for r, i in enumerate(S[ch : ch + 500]):
                    nb = topk_excluding(sc[r], same[i], K)
                    s_lo.append(np.mean(lo[nb] == lo[i]))
                    s_up.append(np.mean(up[nb] == up[i]))
                    if emb_nb is not None:
                        ov.append(len(set(nb) & set(emb_nb[i])) / K)
            ovs = f"{np.mean(ov):.3f}" if ov else "—"
            print(f"| {lang} | {name} | {len(ix.vocab):,} | {ix.nnz:,} | {np.mean(s_lo):.3f} | {np.mean(s_up):.3f} | {ovs} |")
            log(f"  {lang} {name}: {time.time() - t0:.1f} 秒")
            if f is grams_ja or f is grams_en:
                chosen[lang] = ix

    # 2. 近さ別に、同じまとまりに入る割合
    print("\n## 2. 近さ別に、2 件が同じまとまりに入る割合\n")
    _, c_lo = np.unique(lo, return_counts=True)
    _, c_up = np.unique(up, return_counts=True)
    print(f"無作為な 2 件が同じ下位のまとまりに入る割合 {np.sum((c_lo / n) ** 2):.3f}・同じ上位のまとまり {np.sum((c_up / n) ** 2):.3f}\n")
    bins = [0.1, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.7, 1.01]
    for lang, ix in chosen.items():
        cnt = np.zeros(len(bins) - 1)
        s_lo = np.zeros_like(cnt)
        s_up = np.zeros_like(cnt)
        for ch in range(0, len(S), 500):
            sc = (ix.M[S[ch : ch + 500]] @ ix.M.T).toarray()
            for r, i in enumerate(S[ch : ch + 500]):
                s = sc[r]
                s[same[i]] = -1
                b = np.digitize(s, bins) - 1
                ok = (b >= 0) & (b < len(bins) - 1)
                np.add.at(cnt, b[ok], 1)
                np.add.at(s_lo, b[ok], (lo[ok] == lo[i]).astype(float))
                np.add.at(s_up, b[ok], (up[ok] == up[i]).astype(float))
        print(f"### {lang}（目安：少し重なる {LEVELS[lang][0]:.2f}・近い {LEVELS[lang][1]:.2f}・とても近い {LEVELS[lang][2]:.2f}）\n")
        print("| 近さ | 問い 1 件あたりの件数 | 同じ下位のまとまり | 同じ上位のまとまり |")
        print("|---|---|---|---|")
        for k in range(len(bins) - 1):
            if cnt[k]:
                hi = "1" if bins[k + 1] > 1 else f"{bins[k + 1]:.2f}"
                print(f"| {bins[k]:.2f}〜{hi} | {cnt[k] / len(S):.2f} | {s_lo[k] / cnt[k]:.3f} | {s_up[k] / cnt[k]:.3f} |")
        print()

    # 3. 例文
    dump = {}
    XY = np.array([[a["x"], a["y"]] for a in A])
    diag = float(np.linalg.norm(XY.max(0) - XY.min(0)))
    pin_rows = []
    for lang, qs, off in (("ja", QUERIES_JA, OFF_JA), ("en", QUERIES_EN, OFF_EN)):
        if lang not in chosen:
            continue
        ix, L = chosen[lang], LEVELS[lang]
        print(f"## 3. 例文（{lang}）\n")
        tail = "（文末を外さないとき）" if lang == "ja" else ""
        print(f"| 文 | いちばん近い{tail} | {L[0]:.2f} 以上 | {L[1]:.2f} 以上 | {L[2]:.2f} 以上 | {L[1]:.2f} 以上のうち最も多い上位のまとまり | 近い順の 3 件 |")
        print("|---|---|---|---|---|---|---|")
        dump[lang] = {}
        for q in qs + ["—"] + off:
            if q == "—":
                print("| **データに無い話題** | | | | | | |")  # 以下は比べるための文
                continue
            s = ix.scores(prep_ja(q) if lang == "ja" else q)
            order = np.lexsort((np.arange(n), -s))
            top = order[:10]
            dump[lang][q] = [[A[i]["arg_id"], round(float(s[i]), 5)] for i in top]
            near = order[s[order] >= L[1]]
            main_up = ""
            if len(near):
                u, c = np.unique(up[near], return_counts=True)
                k = int(np.argmax(c))
                main_up = f"{clusters[u[k]]['label']}（{c[k]} 件）"
            cut = 28 if lang == "ja" else 48
            t3 = " / ".join(f"{s[i]:.2f} {texts[lang][i][:cut]}" for i in top[:3])
            best = f"{s[top[0]]:.2f}"
            if lang == "ja":  # 問いの文末の「〜してほしい」「です」などを外さずに比べたとき
                best += f"（{ix.scores(norm(q)).max():.2f}）"
            print(f"| {q} | {best} | {(s >= L[0]).sum()} | {(s >= L[1]).sum()} | {(s >= L[2]).sum()} | {main_up} | {t3} |")
            if lang == "ja" and q in qs and len(near):
                h = near[:MARK_K]
                w = s[h]
                cen = (XY[h] * w[:, None]).sum(0) / w.sum()
                gm = geometric_median(XY[h], w)
                d_c = np.linalg.norm(XY[h] - cen, axis=1).min() / diag
                d_g = np.linalg.norm(XY[h] - gm, axis=1).min() / diag
                pin_rows.append((q, len(h), d_c, d_g))
        print()

    # 4. 地図の印
    if pin_rows:
        print("## 4. 地図の印：印から最も近い「近い意見」までの距離（2 次元配置の対角線に対する割合）\n")
        print("| 文 | 使った近い意見 | 重み付き重心 | 重み付き幾何中央値（ページ） |")
        print("|---|---|---|---|")
        for q, k, d_c, d_g in pin_rows:
            print(f"| {q} | {k} | {d_c * 100:.1f}% | {d_g * 100:.1f}% |")
        print()

    print("## 索引の大きさ\n")
    for lang, ix in chosen.items():
        print(f"- {lang}：文字の組 {len(ix.vocab):,} 種・延べ {ix.nnz:,}")
    if args.dump:
        args.dump.write_text(json.dumps(dump, ensure_ascii=False, indent=1), encoding="utf-8")
        log(f"{args.dump} に書き出した")


if __name__ == "__main__":
    main()
