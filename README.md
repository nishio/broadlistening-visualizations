# broadlistening-visualizations

広聴AI（[kouchou-ai](https://github.com/digitaldemocracy2030/kouchou-ai)）の分析結果を、散布図以外の見せ方で描く可視化の試作とカタログ。

ここで試して、使われたものだけを kouchou-ai 本体（公開ビューアの可視化 plugin）に入れるかどうか判断する。本体に入れるかを先に決めずに、まず個人 repo で作って見せる、という置き場。

- デモ：https://nishio.github.io/broadlistening-visualizations/
- 方針の経緯：[kouchou-ai-developer-wiki の open-decisions A13](https://digitaldemocracy2030.github.io/kouchou-ai-developer-wiki/analyses/open-decisions)

## カタログ

| 名前 | 何が見えるか | デモ | 入力 | 状態 | 本体に入れる候補か | 最終確認日 |
|---|---|---|---|---|---|---|
| 意見マンダラート | 意見を 8 つずつに割り、1 マスが意見 1 枚になるまで繰り返した木を、中心＋周り 8 マスの 9×9（スマホは 3×3）で降りていく。最下層から元の提案に飛べる | [mandalart/](https://nishio.github.io/broadlistening-visualizations/mandalart/) | `public_result.json`（＋任意で埋め込み） | 試作 | 未判断 | 2026-11-07 |
| 付箋ズーム | 散布図の点を意見の本文が読める付箋にし、引くとまとまりの表札、寄ると個々の意見が読める | — | `public_result.json` | 未着手（元の試作は別 repo `canvas_kozaneba_prototype`） | 未判断 | 2026-11-07 |
| 階層リスト（ツリー表示） | 上位のクラスタから展開して、最下層で個々の意見と出典まで辿る | 本体の公開ビューア | — | **本体に入った**（kouchou-ai PR #1166、「全部読む」詳細モード） | 済 | 2026-11-07 |

各項目の「最終確認日」は、その可視化が今の入力データと手元の手順で動くことを最後に確かめた日。古くなったら確かめ直すか、状態を書き換える。

## 入力：`public_result.json`

可視化はどれも、広聴AI の公開用 artifact `public_result.json`（kouchou-ai の `docs/development/public-result.md`、schema_version 1）を入力にする。

- 広聴AI で作ったどのレポートでも同じ可視化を描ける
- 本体に入れると決めたとき、同じ入力を読む plugin を書くだけで済む

## 共通の題材：チームみらい 2025 オープンデータ

[nishiohirokazu/team-mirai-2025-embeddings](https://huggingface.co/datasets/nishiohirokazu/team-mirai-2025-embeddings)（CC BY 4.0）。2025 年参院選の「しゃべれるマニフェスト」に届いた政策提案 PR 9,688 件から抽出した意見 21,348 件の広聴AI 分析結果（2 次元配置・20/400 クラスタ・表札・英訳・1536 次元埋め込み）。

出典：チームみらい・しゃべれるマニフェスト参加者 / データセット nishiohirokazu/team-mirai-2025-embeddings（CC BY 4.0）

## 作り方

[uv](https://docs.astral.sh/uv/) を使う。

```sh
# 1. データを取る（--embeddings で 1536 次元の埋め込みも。約 120MB）
uv run scripts/fetch_team_mirai.py --embeddings

# 2. public_result.json の形に変換
uv run scripts/team_mirai_to_public_result.py
# -> data/team-mirai-2025/public_result.json

# 3. マンダラート用の木を作る（約 30 秒）
uv run scripts/build_mandalart_tree.py data/team-mirai-2025/public_result.json \
    --embeddings data/team-mirai-2025/kouchou-ai-embeddings-1536dim.parquet \
    --out docs/mandalart/team-mirai-2025.json

# 4. 見る
cd docs && python3 -m http.server 8000
# -> http://localhost:8000/mandalart/
```

広聴AI で作った自分のレポートで描くときは、出力ディレクトリの `public_result.json` を 3 に渡す（`--embeddings` を省くと 2 次元配置で割る）。ページは `?data=<json のパス>` で別のデータを読める。

## 意見マンダラートの作り

- 広聴AI の既定の階層づくり（最細の段を k-means で切り、Ward で上へ併合する）では、親 1 つあたりの子の数が揃わない（このデータでは上位 20 クラスタの子の数が 1〜38）。マンダラートは中心＋8 マスなので、上から順に k-means（k=8）で割り、子が 8 以下になった節は意見そのものをマスに並べる。深さは枝ごとに違う（このデータで 5,848 節・最大 7 段）
- 埋め込みは長さ 1 に正規化してから割る（cosine）。乱数の種は 42
- 表札は LLM を使わず、各まとまりの中心に最も近い意見をそのまま使う。上の段ほど「〜が不足している」のような一般的な文になりやすいので、上の数段だけ LLM で表札を付ける余地がある

## 本体に入れる判断の目安

- 実際に使われた例がある（会議・発表・自治体での利用など）
- `public_result.json` だけで描ける（足りない項目があれば、その項目を公開契約に足す提案とセットにする）
- 本体側で保守する人がいる

## ライセンス

- コード：MIT（[LICENSE](LICENSE)）
- データ（`docs/*/team-mirai-2025.json`）：CC BY 4.0。出典は上記のとおり
