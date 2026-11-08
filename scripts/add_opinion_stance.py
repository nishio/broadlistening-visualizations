"""賛否スコアを付けた public_result.json を作る（広聴AI 本体の任意ステップ opinion_stance と同じ形）。

本体の opinion_stance は、レポートの question（または target）を「対象」にして、意見 1 件ごとに
LLM へ「対象」と「意見」を渡し、-1.0（反対）〜 +1.0（賛成）の stance_score を arguments[].attributes に入れる。
判定不能（applicable: false）の意見には属性を付けない。プロンプトは本体の既定（STANCE_PROMPT）をそのまま使う
（本体は AGPL-3.0 なので文はここに写さず、--kouchou-ai の checkout か、入っている analysis_core から実行時に読む）。

賛否の対象は 1 つのレポートに 1 つなので、ここでは上位クラスタごとに public_result.json の部分集合を作り、
config.question をその対象に置き換えてから採点する（本体なら、そのまとまりだけを 1 つのレポートとして作るのに当たる）。

    uv run --group llm scripts/add_opinion_stance.py docs/data/team-mirai-2025/public_result.json \\
        --target "1_13=原子力発電を今後も活用すること" --out-dir docs/data/team-mirai-2025-stance \\
        --kouchou-ai ~/kouchou-ai \\
        --endpoints ~/endpoints.local.json --backend qwen --model <モデル名>

出力は <out-dir>/<クラスタ id>/public_result.json（schema_version 1 のまま。属性を足すのは契約の範囲内）。
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from pydantic import BaseModel

sys.path.insert(0, str(Path(__file__).resolve().parent))
from llm_cache import DATA_DIR, add_llm_args, make_llm  # noqa: E402

CACHE_DIR = DATA_DIR / "stance-cache"

# プロンプトは本体の既定（analysis_core/steps/opinion_stance.py の STANCE_PROMPT）を実行時に読む。
# 本体は AGPL-3.0、この repo は MIT なので、プロンプトの文をここに写さない。
STANCE_FILE = "packages/analysis-core/src/analysis_core/steps/opinion_stance.py"


def load_stance_prompt(kouchou_ai: Path | None) -> tuple[str, str]:
    """本体の STANCE_PROMPT を返す。（プロンプト, 出どころ）。kouchou-ai の checkout があればそこから、なければ入っている analysis_core から。"""
    if kouchou_ai:
        f = kouchou_ai / STANCE_FILE if kouchou_ai.is_dir() else kouchou_ai
        for node in ast.parse(f.read_text()).body:
            if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "STANCE_PROMPT" for t in node.targets):
                return ast.literal_eval(node.value), str(f)
        sys.exit(f"{f} に STANCE_PROMPT が見つからない")
    try:
        from analysis_core.steps.opinion_stance import STANCE_PROMPT
    except ImportError:
        sys.exit("本体のプロンプトが読めない：--kouchou-ai に kouchou-ai の checkout を渡すか、kouchou-ai-analysis-core を入れる")
    return STANCE_PROMPT, "analysis_core"


class StanceResponse(BaseModel):
    applicable: bool
    stance_score: float


def subset(public: dict, cid: str, target: str) -> dict:
    """上位クラスタ cid の意見だけを残した public_result.json。config.question を賛否の対象にする。"""
    args = [a for a in public["arguments"] if len(a["cluster_ids"]) > 1 and a["cluster_ids"][1] == cid]
    used = {c for a in args for c in a["cluster_ids"]}
    clusters = [c for c in public["clusters"] if c["id"] in used]
    arg_ids = {a["arg_id"] for a in args}
    comment_ids = {a["comment_id"] for a in args}
    label = next(c["label"] for c in clusters if c["id"] == cid)
    tr = {}
    for lang, t in (public.get("translations") or {}).items():
        tr[lang] = {
            "clusters": {k: v for k, v in (t.get("clusters") or {}).items() if k in used},
            "arguments": {k: v for k, v in (t.get("arguments") or {}).items() if k in arg_ids},
        }
    config = dict(public.get("config") or {})
    config["intro"] = f"「{config.get('question', '')}」のうち、まとまり「{label}」の意見 {len(args):,} 件。" + (config.get("intro") or "")
    config["question"] = target
    return {
        **public,
        "arguments": [json.loads(json.dumps(a)) for a in args],
        "clusters": clusters,
        "comments": {k: v for k, v in (public.get("comments") or {}).items() if k in comment_ids},
        "propertyMap": {k: {i: v for i, v in m.items() if i in arg_ids} for k, m in (public.get("propertyMap") or {}).items()},
        "translations": tr,
        "overview": f"まとまり「{label}」の意見に、「{target}」への賛否スコアを付けたもの。",
        "comment_num": len(comment_ids),
        "config": config,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("public_result", type=Path)
    ap.add_argument("--target", action="append", required=True, help="クラスタ id=賛否の対象（くり返し指定できる）")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--kouchou-ai", type=Path, help="kouchou-ai の checkout（または opinion_stance.py）。本体の既定プロンプトをここから読む")
    add_llm_args(ap)
    args = ap.parse_args()

    public = json.loads(args.public_result.read_text())
    prompt, prompt_src = load_stance_prompt(args.kouchou_ai)
    print(f"プロンプト：{prompt_src}（sha256 {hashlib.sha256(prompt.encode()).hexdigest()[:12]}）", file=sys.stderr)
    llm, workers, base_url = make_llm(args, CACHE_DIR)
    pool = ThreadPoolExecutor(max_workers=workers)
    for spec in args.target:
        cid, target = spec.split("=", 1)
        sub = subset(public, cid.strip(), target.strip())
        if not sub["arguments"]:
            sys.exit(f"クラスタ {cid} が見つからない")
        texts = [a["argument"] for a in sub["arguments"]]
        res = list(pool.map(lambda t: llm.parse("stance", prompt, f"対象: {target.strip()}\n意見: {t}", StanceResponse), texts))
        if args.dry_run:
            continue
        bins = Counter()
        for a, r in zip(sub["arguments"], res):
            s = max(-1.0, min(1.0, float(r.stance_score)))
            if r.applicable:
                a.setdefault("attributes", {})["stance_score"] = round(s, 3)
                bins["strongly_against" if s <= -0.6 else "against" if s < -0.2 else "neutral" if s <= 0.2 else "for" if s < 0.6 else "strongly_for"] += 1
            else:
                bins["not_applicable"] += 1
        out = args.out_dir / cid.strip() / "public_result.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(sub, ensure_ascii=False, separators=(",", ":")))
        print(f"[{cid}] {target.strip()}：{len(texts)} 件 {dict(bins)} → {out}", file=sys.stderr)
    print(f"呼び出し {dict(llm.calls)}・API {dict(llm.fresh)}・応答待ち {llm.seconds:.0f} 秒（並列 {workers}）", file=sys.stderr)


if __name__ == "__main__":
    main()
