# broadlistening-visualizations

広聴AI（kouchou-ai）の分析結果を、散布図以外の見せ方で描く可視化の試作とカタログ。nishio の個人 repo。カタログ・作り方・各可視化の仕組みは README.md が正本なので、まずそれを読む。このファイルは作業のルールだけを書く。

## 方針

- **入力は `public_result.json` だけ**にする（schema_version 1。仕様は kouchou-ai の `docs/development/public-result.md`。手元では `/home/ubuntu/kouchou-ai-developer-wiki/work/kouchou-ai/` にある）。本体に入れると決めたとき、同じ入力を読む plugin を書くだけで済むようにするため
  - 題材ごとの事情（チームみらいのデータの列名など）は `scripts/` の変換スクリプトに閉じ込める。可視化の側に持ち込まない
  - `public_result.json` に足りない項目が要るときは、黙って別の入力を読まない。どの項目が足りないかを書き出し、公開契約に足す提案として nishio に上げる
  - 埋め込みのような任意の追加入力を使うときは、それなしでも動く道を残す（例：マンダラートは `--embeddings` を省くと 2 次元配置で割る）
- **重い計算は Python で事前に済ませ、ブラウザ側は読んで描くだけ**にする（マンダラートの木、付箋ズームの配置と出始める拡大率）
- 可視化のページはビルドなしの静的 HTML 1 枚。今は外部ライブラリを読んでいない。足すなら理由を README に書く
- どのページも `?data=<URL>` で別のレポートの JSON を読めるようにする
- **元になったものがあるときは、作る前にそれを読む**。付箋ズームの最初の版はオリジナル（nishio の別 repo `canvas_kozaneba_prototype` の `StickyNotesZoomDemo`）を読まずに作って、拡大すると付箋が震えた。階層リストは本体の PR #1166 が元

## 置き場

- `scripts/`：データ取得・変換・事前計算。`uv run` で動かす
- `docs/`：GitHub Pages の公開元（main の `/docs`、legacy build）。**main に push するとそのまま公開される**
- `docs/data/`・`docs/<可視化>/*.json`：公開する生成物。コミットする（データは CC BY 4.0。出典表記を消さない）
- `data/`：取得した元データと埋め込み（約 120MB）。gitignore 済み。コミットしない。GitHub は 1 ファイル 100MB が上限なので、生成物が大きくなったら分割か間引きを考える

## 新しい可視化を足すとき

1. `docs/<名前>/index.html` を作る。事前計算が要るなら `scripts/build_<名前>_*.py` を足す
2. README のカタログ表に行を足し、「作り方」に手順を、必要なら「〜の作り」節に仕組みを書く。`docs/index.html` の一覧にも足す
3. `cd docs && python3 -m http.server 8000` でブラウザで確かめる。拡大・展開・検索など、その可視化の肝の操作を実際に動かす。確かめた数値（枚数・重なり・段数など）は README に残す
4. 「最終確認日」は、実際に動くのを確かめた日だけ更新する

## 作業の流れ

- 返答・途中経過・コミットメッセージは日本語
- main で作業する。commit / push は nishio の指示があるときにする（push は公開と同じ）
- 本体（kouchou-ai）に入れるかどうかは nishio が決める。README の「本体に入れる判断の目安」は判断材料で、満たしても AI から本体に PR や Issue を出さない

## kouchou-ai-developer-wiki への書き戻し

区切りごとに、`/home/ubuntu/kouchou-ai-developer-wiki` に結果を書き戻す。あちらの CLAUDE.md に従う（main で作業する・別セッションが同時に触っている・公開 wiki なので公開境界を守る・本文の #NNN は GitHub へのリンクにする）。

- `wiki/sources/team-mirai-2025-open-dataset.md` の「個人 repo のカタログ」節：何を作ったか・どう作ったか・確かめた結果。`## Updates` に日付つきで 1 行
- `wiki/analyses/open-decisions.md` の A13：方針や本体に入れる判断が動いたときだけ
- `wiki/log.md` の先頭に `## [YYYY-MM-DD HH:MM] filing-back | ...` を足し、`python3 scripts/refresh_logs.py`
- 定例で話す価値があれば `wiki/concepts/meeting-report-draft.md` にも

## 残っている論点（2026-11-07 時点）

- マンダラートの上の段の表札を LLM で付けるか、代表意見のままにするか
- 本体 docs の関連研究ページ（kouchou-ai PR #1174）からこのカタログへのリンクの張り方
