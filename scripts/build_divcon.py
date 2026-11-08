"""対立軸ビュー（DivCon）用の事前計算：まとまりごとに対立軸を見つけ、各意見を軸の上で 6 段階に採点する。

元は tokoroten の DivCon（https://github.com/tokoroten/divcon 、MIT、experiments/divcon_analysis.py）。
プロンプトと段の構成はそれに従い、次の点だけ変えている。

- DivCon の Stage 1・2（LLM によるトピック発見と意見の分類）は行わず、広聴AI の上位クラスタをそのままトピックにする。
  本体に載せるときも「クラスタごとの対立軸」として既存の結果につなげられるように
- 対象のクラスタは --clusters で選ぶ（賛否の割れる話題だけを選んで試すため）
- 入力の意見は extraction 後の短い文なので、Stage 4 の excerpt（本文の切り抜き）は求めず、短い理由だけを返させる
- LLM への ID は意見の arg_id ではなくクラスタ内の通し番号にする（トークンの節約と取り違え防止）

段
- 3a 対立軸の発見：クラスタの意見（最大 --sample 件を無作為に）から 2〜4 本の「A vs B」を見つけ、強度 1〜5 を付ける
- 3b アンカーの生成：軸ごとに、左極・右極それぞれの極端な主張を 10 個ずつ生成する（実在の意見ではない）
- 4  採点：アンカーを基準に、クラスタの全意見を 1（左極）〜6（右極）で採点。軸に当たらない意見は null
- 5  合意可能性：左右の意見から、合意できそうな点と根本的に対立する点を挙げる（DivCon の「Conciliate」）

LLM の応答は data/divcon-cache/ に保存し、同じ呼び出しは再利用する（途中で落ちても続きから再開でき、課金し直さない）。
既定は OpenAI の API（キーは環境変数 OPENAI_API_KEY、なければ data/.env（gitignore 済み）の OPENAI_API_KEY=...）。
--base-url か --endpoints/--backend で OpenAI 互換の接続先（手元や組織内の LLM）も使える。共有の接続先では並列を 2 に抑える。

    # 呼び出し回数の見積もりだけ（API は呼ばない）
    uv run --group llm scripts/build_divcon.py docs/data/team-mirai-2025/public_result.json \\
        --clusters 1_13,1_1,1_20,1_19 --dry-run

    uv run --group llm scripts/build_divcon.py docs/data/team-mirai-2025/public_result.json \\
        --clusters 1_13,1_1,1_20,1_19 --out docs/divcon/team-mirai-2025.json

    # OpenAI 互換の接続先で（接続先は追跡しない JSON に書く）
    uv run --group llm scripts/build_divcon.py ... --endpoints ~/endpoints.local.json --backend qwen \\
        --model <接続先のモデル名> --insecure --out docs/divcon/team-mirai-2025.json
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import date
from pathlib import Path
from typing import List, Optional

from pydantic import BaseModel

sys.path.insert(0, str(Path(__file__).resolve().parent))
from llm_cache import DATA_DIR, LLM, add_llm_args, make_llm  # noqa: E402

CACHE_DIR = DATA_DIR / "divcon-cache"


# ---------------------------------------------------------------- 応答の型（DivCon と同じ形。Score だけ excerpt を外した）


class Axis(BaseModel):
    id: str
    name: str
    left_pole: str
    right_pole: str
    strength: int
    reasoning: str


class AxisDiscoveryResponse(BaseModel):
    axes: List[Axis]


class AnchorGenerationResponse(BaseModel):
    left_anchors: List[str]
    right_anchors: List[str]


class Score(BaseModel):
    opinion_id: str
    score: Optional[int] = None  # 1〜6。軸に当たらなければ null
    reasoning: str


class ScoringResponse(BaseModel):
    scores: List[Score]


class ConsensusPoint(BaseModel):
    point: str
    explanation: str
    supporting_opinions: List[str]


class ConflictPoint(BaseModel):
    point: str
    explanation: str
    left_opinions: List[str]
    right_opinions: List[str]


class ConsensusAnalysisResponse(BaseModel):
    consensus_points: List[ConsensusPoint]
    conflict_points: List[ConflictPoint]
    reasoning: str


# ---------------------------------------------------------------- 段（プロンプトは DivCon のもの。Stage 4 の excerpt の指示だけ外した）


def opinions_text(ops: list[tuple[str, str]]) -> str:
    return "\n\n".join(f"[{i}] {t}" for i, t in ops)


def stage3a(llm: LLM, topic: str, sample: list[tuple[str, str]], cid: str) -> list[dict]:
    prompt = f"""以下は、「{topic}」というトピックに関する市民意見です。

{opinions_text(sample)}

【タスク】
この意見群の中で、人々が**対立している軸**を発見してください。

対立軸とは:
- 意見が二極化している次元
- 「AかBか」という選択を迫られる論点
- 例: "環境保護 vs 経済成長", "短期的利益 vs 長期的持続性"

【指示】
1. 主要な対立軸を2-4個抽出してください
2. 各軸に明確な名前を付けてください（"A vs B"の形式）
3. 対立の強度を5段階で評価してください：
   - 5: 非常に強い対立（意見が完全に二極化）
   - 4: 強い対立
   - 3: 中程度の対立
   - 2: やや対立
   - 1: 弱い対立
4. **重要**: reasoning（理由説明）で具体的な意見を参照する際は、必ず `[ID:XXX]` の形式を使用してください
   - 正しい例: "[ID:123]では再エネ推進を主張している"
   - 正しい例: "左極の代表例として[ID:456]、右極として[ID:789]が見られる"
   - 誤った例: "意見123では..." "ID123によると..." "コメント[123]"
"""
    r = llm.parse("3a", "あなたは対立構造を分析する専門家です。", prompt, AxisDiscoveryResponse)
    if r is None:
        return [{"id": f"{cid}_A{k + 1}", "name": "(dry-run)", "left_pole": "", "right_pole": "", "strength": 0, "reasoning": ""} for k in range(3)]
    axes = []
    for k, a in enumerate(r.axes):
        d = a.model_dump()
        d["id"] = f"{cid}_A{k + 1}"  # LLM の付けた id は使わず、クラスタ id から振り直す
        d["strength"] = max(1, min(5, int(d["strength"])))
        axes.append(d)
    return axes


def stage3b(llm: LLM, axis: dict, sample: list[tuple[str, str]]) -> dict:
    prompt = f"""以下の市民意見を参考にして、対立軸「{axis['name']}」について、
極端に強い主張の文章例を生成してください。

【既存の意見（参考）】
{opinions_text(sample)}

【タスク】
- **左極**（{axis['left_pole']}）の極端な主張を10パターン生成
- **右極**（{axis['right_pole']}）の極端な主張を10パターン生成

【極端な文章の例】
例えば、対立軸が「環境保護 vs 経済成長」の場合：

左極（環境保護）の例:
- 「経済成長を完全に停止してでも、環境保護を最優先すべきである」
- 「全ての企業活動を即座に規制し、自然環境を元の状態に戻すべきだ」
- 「人間の経済活動は地球環境に対する犯罪であり、全面的に見直すべきである」

右極（経済成長）の例:
- 「環境規制を全て撤廃し、経済成長を最大化すべきである」
- 「環境保護は経済発展の後で考えればよく、今は成長が最優先だ」
- 「環境コストを無視してでも、産業競争力を強化すべきである」

【重要なポイント】
1. 既存の意見の文脈を保ちつつ、より極端な表現にすること
2. 「完全に」「全て」「絶対に」「即座に」などの強い表現を使用
3. 妥協や条件を一切含まない断定的な主張にすること
4. 各パターンは異なる角度から極端さを表現すること
5. 同じような主張の繰り返しは避けること
"""
    r = llm.parse("3b", "多様で極端な主張を生成してください。妥協のない、断定的な表現を使用してください。", prompt, AnchorGenerationResponse, effort="high")
    if r is None:
        return {"left_anchors": [], "right_anchors": []}
    return {"left_anchors": r.left_anchors, "right_anchors": r.right_anchors}


def stage4_batch(llm: LLM, axis: dict, anchors: dict, batch: list[tuple[str, str]]) -> list[dict] | None:
    left = "\n".join(f"L{i + 1}. {a}" for i, a in enumerate(anchors["left_anchors"]))
    right = "\n".join(f"R{i + 1}. {a}" for i, a in enumerate(anchors["right_anchors"]))
    prompt = f"""以下の基準アンカーに基づいて、意見をスコアリングしてください。

【対立軸】{axis['name']}
- 左極（スコア1）: {axis['left_pole']}
- 右極（スコア6）: {axis['right_pole']}

【左極アンカー例】（スコア1に相当）
{left}

【右極アンカー例】（スコア6に相当）
{right}

【スコアリング対象の意見】
{opinions_text(batch)}

【タスク】
各意見を以下の基準でスコアリングしてください:

**まず、この対立軸に該当するかを判定:**
- 意見がこの対立軸について明確な立場を示している場合 → 1-6でスコアリング
- 意見がこの対立軸に全く言及していない、または判断できない場合 → scoreをnullにする

**スコアの意味（該当する場合）:**
6段階評価により、より明確な立場判定を行います。中立的なバランス点はありません。
- **1**: 左極（最も強い）- 左極アンカーに非常に近い立場
- **2**: 左寄り（強）- 左極に近いが、若干の留保がある
- **3**: 左寄り（弱）- 左寄りだが、やや穏健な立場
- **4**: 右寄り（弱）- 右寄りだが、やや穏健な立場
- **5**: 右寄り（強）- 右極に近いが、若干の留保がある
- **6**: 右極（最も強い）- 右極アンカーに非常に近い立場

**reasoning（理由）:** 判断の理由を40字以内で簡潔に書いてください。scoreがnullの場合は空文字列（""）にしてください。

opinion_id には各意見の [ ] 内の番号をそのまま書き、全ての意見について1件ずつ返してください。

**重要**: 意見が対立軸に該当しない場合、無理にスコアを付けず、scoreフィールドをnullにしてください。
"""
    r = llm.parse("4", "アンカーを基準に意見をスコアリングしてください。", prompt, ScoringResponse)
    return None if r is None else [s.model_dump() for s in r.scores]


def stage5(llm: LLM, axis: dict, left: list[tuple[str, int, str]], right: list[tuple[str, int, str]]) -> dict | None:
    lt = "\n".join(f"[ID: {i}, スコア: {s}] {t}" for i, s, t in left)
    rt = "\n".join(f"[ID: {i}, スコア: {s}] {t}" for i, s, t in right)
    prompt = f"""以下の対立軸において、合意可能なポイントと合意不可能なポイントを分析してください。

【対立軸】
{axis['name']}

- 左極（スコア1-3）: {axis['left_pole']}
- 右極（スコア4-6）: {axis['right_pole']}

【スコアリング基準】
- スコア1: 左極の立場を最も強く支持（最左端）
- スコア2-3: 左寄りの立場
- スコア4-5: 右寄りの立場
- スコア6: 右極の立場を最も強く支持（最右端）

【左側の意見（スコア1-3）】
{lt if lt else '（該当意見なし）'}

【右側の意見（スコア4-6）】
{rt if rt else '（該当意見なし）'}

【分析指示】
1. **合意可能なポイント**: 左右両側が妥協・合意できそうな共通点や中間案を特定してください
   - 両側が共有している価値観や懸念
   - 実現可能な妥協案や段階的アプローチ
   - 両側が受け入れられそうな条件付き合意
   - 特にスコア2-3と4-5の意見間で見られる共通点に注目

2. **合意不可能なポイント**: 根本的に対立しており、妥協が困難なポイントを特定してください
   - 価値観の根本的な相違
   - 互いに譲れない原則や立場
   - 論理的に両立不可能な主張
   - 特にスコア1と6の意見間で見られる根本的対立に注目

各ポイントには、それを裏付ける意見のIDを含めてください（supporting_opinions / left_opinions / right_opinions には ID の番号だけを書いてください）。
"""
    r = llm.parse("5", "あなたは対立する意見を分析し、合意可能性を評価する専門家です。", prompt, ConsensusAnalysisResponse)
    return None if r is None else r.model_dump()


# ---------------------------------------------------------------- 全体


ID_REF = re.compile(r"[\[（(]?\s*ID[:：]?\s*(\d+(?:\s*[,，、]\s*\d+)*)\s*[\]）)]?")


def link_ids(text: str, arg_of: dict[str, str]) -> str:
    """説明文の「[ID:12]」「ID 3, 5」などのクラスタ内の通し番号を ⟦arg_id,arg_id⟧ に置き換える（表示側で意見へのリンクにする）。"""
    def rep(m: re.Match) -> str:
        ids = [arg_of[x] for x in re.split(r"\s*[,，、]\s*", m.group(1)) if x in arg_of]
        return "⟦" + ",".join(ids) + "⟧" if ids else ""
    return ID_REF.sub(rep, text or "")


def pick_side(items: list[tuple[str, int, str]], strong: set[int], rng: random.Random, n: int = 15) -> list[tuple[str, int, str]]:
    """合意可能性の分析に渡す意見：強い立場と穏健な立場から半分ずつ無作為に。"""
    s = [x for x in items if x[1] in strong]
    m = [x for x in items if x[1] not in strong]
    rng.shuffle(s)
    rng.shuffle(m)
    out = s[:n] + m[:n]
    if len(out) < 2 * n:
        out += (s[n:] + m[n:])[: 2 * n - len(out)]
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("public_result", type=Path)
    ap.add_argument("--clusters", required=True, help="トピックにする上位クラスタの id（カンマ区切り）")
    ap.add_argument("--out", type=Path)
    add_llm_args(ap)
    ap.add_argument("--effort", default="medium", help="reasoning_effort（アンカー生成だけは DivCon と同じく high）")
    ap.add_argument("--sample", type=int, default=500, help="対立軸の発見・アンカー生成に渡す意見の上限")
    ap.add_argument("--batch", type=int, default=20, help="採点 1 回あたりの意見数")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    public = json.loads(args.public_result.read_text())
    clusters = {c["id"]: c for c in public["clusters"]}
    by_cluster: dict[str, list[dict]] = defaultdict(list)
    for a in public["arguments"]:
        if len(a["cluster_ids"]) > 1:
            by_cluster[a["cluster_ids"][1]].append(a)
    cids = [c.strip() for c in args.clusters.split(",") if c.strip()]
    for cid in cids:
        if cid not in clusters or not by_cluster.get(cid):
            sys.exit(f"クラスタ {cid} が見つからない")

    llm, workers, base_url = make_llm(args, CACHE_DIR, effort=args.effort)
    pool = ThreadPoolExecutor(max_workers=workers)
    topics, axes_out, scores_out = [], [], {}

    for cid in cids:
        args_c = by_cluster[cid]
        local = [(str(i), a["argument"]) for i, a in enumerate(args_c)]  # LLM に渡す ID はクラスタ内の通し番号
        arg_of = {str(i): a["arg_id"] for i, a in enumerate(args_c)}
        text_of = dict(local)
        rng = random.Random(f"{args.seed}:{cid}")
        sample = local if len(local) <= args.sample else rng.sample(local, args.sample)
        topic = clusters[cid]["label"]
        topics.append({"cluster_id": cid, "label": topic, "n": len(local)})
        print(f"[{cid}] {topic}（{len(local)} 件）", file=sys.stderr)

        axes = stage3a(llm, topic, sample, cid)
        anchors_list = list(pool.map(lambda ax: stage3b(llm, ax, sample), axes))
        for ax, anchors in zip(axes, anchors_list):
            scored: dict[str, tuple[int | None, str]] = {}
            todo = local
            for _ in range(2):  # 応答から漏れた意見は、漏れた分だけでもう 1 回採点する
                batches = [todo[i : i + args.batch] for i in range(0, len(todo), args.batch)]
                results = list(pool.map(lambda b: stage4_batch(llm, ax, anchors, b), batches))
                for b, res in zip(batches, results):
                    if res is None:
                        continue
                    want = {i for i, _ in b}
                    for s in res:
                        oid = str(s["opinion_id"]).strip("[] ")
                        if oid in want and oid not in scored:
                            sc = s["score"]
                            scored[oid] = (sc if isinstance(sc, int) and 1 <= sc <= 6 else None, s["reasoning"])
                todo = [x for x in local if x[0] not in scored]
                if not todo or args.dry_run:
                    break
            missing = 0 if args.dry_run else len(todo)
            rows = [[arg_of[i], sc, why] for i, (sc, why) in sorted(scored.items(), key=lambda kv: int(kv[0])) if sc is not None]
            n_na = sum(1 for sc, _ in scored.values() if sc is None)
            scores_out[ax["id"]] = rows
            consensus = None
            if not args.dry_run:
                items = [(i, sc, text_of[i]) for i, (sc, _) in scored.items() if sc is not None]
                left = pick_side([x for x in items if x[1] <= 3], {1}, rng)
                right = pick_side([x for x in items if x[1] >= 4], {6}, rng)
                consensus = stage5(llm, ax, left, right)
                if consensus:  # 意見の参照を通し番号から arg_id へ戻す（説明文の中の参照も）
                    consensus["reasoning"] = link_ids(consensus["reasoning"], arg_of)
                    for p in consensus["consensus_points"] + consensus["conflict_points"]:
                        p["explanation"] = link_ids(p["explanation"], arg_of)
                    fix = lambda ids: [arg_of[i] for i in (str(x).strip("[]ID: ") for x in ids) if i in arg_of]
                    for p in consensus["consensus_points"]:
                        p["supporting_opinions"] = fix(p["supporting_opinions"])
                    for p in consensus["conflict_points"]:
                        p["left_opinions"], p["right_opinions"] = fix(p["left_opinions"]), fix(p["right_opinions"])
            else:
                llm.calls["5"] += 1
            ax["reasoning"] = link_ids(ax["reasoning"], arg_of)
            dist = Counter(r[1] for r in rows)
            axes_out.append({**ax, "cluster_id": cid, **anchors, "n_scored": len(rows), "n_na": n_na, "n_missing": missing,
                             "dist": [dist.get(k, 0) for k in range(1, 7)], "consensus": consensus})
            print(f"  {ax['id']} {ax['name']}：採点 {len(rows)}・該当なし {n_na}・応答漏れ {missing}・分布 {[dist.get(k, 0) for k in range(1, 7)]}", file=sys.stderr)

    print(f"呼び出し（キャッシュ命中を含む）: {dict(llm.calls)} 計 {sum(llm.calls.values())}", file=sys.stderr)
    print(f"API を実際に呼んだ回数: {dict(llm.fresh)}・トークン {dict(llm.usage)}・応答待ちの合計 {llm.seconds:.0f} 秒（並列 {workers}）", file=sys.stderr)
    if args.dry_run:
        print("（dry-run：対立軸は 1 クラスタ 3 本と仮定して見積もった。実際は 2〜4 本）", file=sys.stderr)
        return
    if args.out:
        out = {
            "meta": {
                "method": "DivCon (tokoroten/divcon) の Stage 3a〜5。トピックは広聴AI の上位クラスタ",
                "source": "https://github.com/tokoroten/divcon",
                "model": args.model, "effort": None if base_url else args.effort, "sample": args.sample, "batch": args.batch, "seed": args.seed,
                "built": date.today().isoformat(), "n_arguments": sum(t["n"] for t in topics),
            },
            "topics": topics,
            "axes": axes_out,
            "scores": scores_out,
        }
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(out, ensure_ascii=False, separators=(",", ":")))
        print(f"wrote {args.out} ({args.out.stat().st_size / 1e6:.2f} MB)", file=sys.stderr)


if __name__ == "__main__":
    main()
