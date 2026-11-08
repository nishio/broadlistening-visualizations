"""LLM 呼び出し（キャッシュつき）。対立軸ビューと賛否スライダーの事前計算で共有する。

- OpenAI の API か、OpenAI 互換の接続先（手元や組織内の LLM）を呼ぶ
- 応答はキャッシュ用ディレクトリ（data/ 以下、gitignore 済み）に保存し、同じ呼び出しは再利用する。
  途中で落ちても続きから再開でき、課金し直さない
- 接続先のホスト名は追跡しない JSON（--endpoints）から読み、repo に入れない
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
from collections import Counter
from pathlib import Path

from pydantic import BaseModel

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def load_env_file(path: Path) -> None:
    """data/.env（gitignore 済み）の KEY=VALUE を環境変数に入れる。値は表示しない。"""
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip().removeprefix("export ").strip(), v.strip().strip('"').strip("'"))


def extract_json(text: str) -> str:
    """応答から JSON の部分を取り出す（思考の <think> や ``` の囲みを外す）。"""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S)
    m = re.search(r"```(?:json)?\s*(.*?)```", text, re.S)
    if m:
        text = m.group(1)
    starts = [p for p in (text.find("{"), text.find("[")) if p >= 0]
    return text[min(starts):].strip() if starts else text.strip()


class LLM:
    """OpenAI の API か、OpenAI 互換の接続先（base_url）を呼ぶ。応答は cache_dir に保存して再利用する。

    OpenAI のときは Structured Outputs（DivCon と同じ）。互換の接続先では JSON Schema をプロンプトに書き、
    返ってきた JSON を pydantic で検証する。壊れていれば呼び直す。
    """

    def __init__(self, cache_dir: Path, model: str, effort: str, dry_run: bool, base_url: str | None = None, insecure: bool = False):
        self.cache_dir = cache_dir
        self.model, self.effort, self.dry_run, self.base_url = model, effort, dry_run, base_url
        self.calls = Counter()  # 段ごとの呼び出し回数（キャッシュ命中を含む）
        self.fresh = Counter()  # 実際に API を呼んだ回数
        self.usage = Counter()
        self.seconds = 0.0
        self.client = None
        if not dry_run:
            from openai import OpenAI

            if base_url:
                kw = {}
                if insecure:  # openai 3.x は httpx2、2.x 以前は httpx を使う
                    try:
                        import httpx2 as httpx
                    except ImportError:
                        import httpx
                    kw["http_client"] = httpx.Client(verify=False, timeout=600)
                self.client = OpenAI(base_url=base_url, api_key=os.environ.get("OPENAI_COMPAT_API_KEY", "EMPTY"), timeout=600, **kw)
            else:
                if not os.environ.get("OPENAI_API_KEY"):
                    load_env_file(DATA_DIR / ".env")
                if not os.environ.get("OPENAI_API_KEY"):
                    sys.exit("OPENAI_API_KEY が環境変数にも data/.env にもない（--dry-run なら不要）")
                self.client = OpenAI()
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _call_openai(self, system: str, user: str, schema: type[BaseModel], effort: str):
        c = self.client.chat.completions.parse(
            model=self.model,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            reasoning_effort=effort,
            response_format=schema,
        )
        return c, c.choices[0].message.parsed

    def _call_compat(self, system: str, user: str, schema: type[BaseModel], temperature: float = 0.0):
        # response_format の JSON Schema（制約つき生成）は、試した接続先で空白を延々と出し続けて上限まで止まらなかったので使わない
        js = json.dumps(schema.model_json_schema(), ensure_ascii=False)
        user = user + "\n\n【出力形式】次の JSON Schema に従う JSON だけを出力してください。\n" + js
        c = self.client.chat.completions.create(
            model=self.model, temperature=temperature, max_tokens=8000,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}])
        if c.choices[0].finish_reason == "length":
            raise ValueError("出力が上限で切れた")
        # 文字列の中に生の改行などが混じることがあるので strict=False で読む
        return c, schema.model_validate(json.loads(extract_json(c.choices[0].message.content or ""), strict=False))

    def parse(self, stage: str, system: str, user: str, schema: type[BaseModel], effort: str | None = None):
        effort = effort or self.effort
        self.calls[stage] += 1
        key = hashlib.sha256(json.dumps([self.base_url and "compat", self.model, effort, schema.__name__, system, user], ensure_ascii=False).encode()).hexdigest()
        path = self.cache_dir / f"{key}.json"
        if path.exists():
            return schema.model_validate_json(json.loads(path.read_text())["response"])
        if self.dry_run:
            return None
        t0 = time.time()
        for attempt in range(4):
            try:
                # 温度 0 だと同じ壊れ方を繰り返すので、呼び直しでは温度を上げる
                c, parsed = (self._call_compat(system, user, schema, 0.0 if attempt == 0 else 0.4) if self.base_url
                             else self._call_openai(system, user, schema, effort))
                break
            except Exception as e:  # 通信・壊れた JSON など。本文やキー・接続先は出さない
                if attempt == 3:
                    raise
                print(f"  [retry {attempt + 1}] {stage}: {type(e).__name__}", file=sys.stderr)
                time.sleep(5 * (attempt + 1))
        self.seconds += time.time() - t0
        self.fresh[stage] += 1
        if c.usage:
            self.usage["input"] += c.usage.prompt_tokens
            self.usage["output"] += c.usage.completion_tokens
        path.write_text(json.dumps({"stage": stage, "model": self.model, "response": parsed.model_dump_json()}, ensure_ascii=False))
        return parsed




def add_llm_args(ap: argparse.ArgumentParser, default_workers_openai: int = 8) -> None:
    ap.add_argument("--model", default=os.environ.get("OPENAI_MODEL", "gpt-5-mini"))
    ap.add_argument("--base-url", help="OpenAI 互換の接続先（例 https://host/v1）。指定すると OpenAI ではなくこちらを呼ぶ")
    ap.add_argument("--endpoints", type=Path, help="接続先を書いた追跡しない JSON（{名前: base_url}）。--backend と組で使い、ホスト名を repo に入れない")
    ap.add_argument("--backend", help="--endpoints の中の名前")
    ap.add_argument("--insecure", action="store_true", help="接続先の TLS 証明書を検証しない（社内の自己署名証明書など）")
    ap.add_argument("--workers", type=int, help=f"並列数（既定は OpenAI で {default_workers_openai}、互換の接続先で 2。共有の接続先では上げない）")
    ap.add_argument("--dry-run", action="store_true", help="API を呼ばず、呼び出し回数だけ見積もる")


def make_llm(args: argparse.Namespace, cache_dir: Path, effort: str = "medium", default_workers_openai: int = 8) -> tuple[LLM, int, str | None]:
    """引数から LLM を作る。（LLM, 並列数, 互換の接続先の base_url か None）を返す。"""
    base_url = args.base_url
    if args.endpoints:
        if not args.backend:
            sys.exit("--endpoints には --backend が要る")
        base_url = json.loads(args.endpoints.expanduser().read_text())[args.backend]
    workers = args.workers or (2 if base_url else default_workers_openai)
    return LLM(cache_dir, args.model, effort, args.dry_run, base_url=base_url, insecure=args.insecure), workers, base_url
