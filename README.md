# broadlistening-visualizations

広聴AI（[kouchou-ai](https://github.com/digitaldemocracy2030/kouchou-ai)）の分析結果を、散布図以外の見せ方で描く可視化の試作とカタログ。

ここで試して、使われたものだけを kouchou-ai 本体（公開ビューアの可視化 plugin）に入れるかどうか判断する。本体に入れるかを先に決めずに、まず個人 repo で作って見せる、という置き場。

- デモ：https://nishio.github.io/broadlistening-visualizations/
- 方針の経緯：[kouchou-ai-developer-wiki の open-decisions A13](https://digitaldemocracy2030.github.io/kouchou-ai-developer-wiki/analyses/open-decisions)

## カタログ

| 名前 | 何が見えるか | デモ | 入力 | 状態 | 本体に入れる候補か | 最終確認日 |
|---|---|---|---|---|---|---|
| 意見マンダラート | 意見を 8 つずつに割り、1 マスが意見 1 枚になるまで繰り返した木を、中心＋周り 8 マスで降りていく（既定は 3×3、9×9 で周り 8 マスの中身も見える）。最下層から元の提案に飛べる | [mandalart/](https://nishio.github.io/broadlistening-visualizations/mandalart/) | `public_result.json`（＋任意で埋め込み）から作った木 | 試作 | 未判断 | 2026-11-08 |
| 付箋ズーム | 2 次元配置の上に意見を付箋で置く。引いた状態でも各まとまりの表札が付箋で読め、寄るほど意見の付箋が増え、最後は全意見が付箋になる。付箋の位置と出始める拡大率は事前計算なので、拡大しても入れ替わらない | [sticky/](https://nishio.github.io/broadlistening-visualizations/sticky/) | `public_result.json` から作った配置 | 試作（全意見を格子に並べる部分は別 repo `canvas_kozaneba_prototype` の方式） | 未判断 | 2026-11-08 |
| 階層リスト（ツリー表示） | 上位のクラスタから展開して、最下層で個々の意見と出典まで辿る。検索つき | [tree/](https://nishio.github.io/broadlistening-visualizations/tree/) | `public_result.json` | **本体に入った**（kouchou-ai PR #1166、「全部読む」詳細モード）。ここは単体版 | 済 | 2026-11-07 |

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

# 2. public_result.json の形に変換（付箋ズームと階層リストはこれをそのまま読む）
uv run scripts/team_mirai_to_public_result.py --out docs/data/team-mirai-2025/public_result.json

# 3. マンダラート用の木を作る（約 30 秒）
uv run scripts/build_mandalart_tree.py docs/data/team-mirai-2025/public_result.json \
    --embeddings data/team-mirai-2025/kouchou-ai-embeddings-1536dim.parquet \
    --out docs/mandalart/team-mirai-2025.json

# 4. 付箋ズーム用の配置を作る（約 10 秒）
uv run scripts/build_sticky_layout.py docs/data/team-mirai-2025/public_result.json \
    --out docs/sticky/team-mirai-2025.layout.json

# 5. 見る
cd docs && python3 -m http.server 8000
# -> http://localhost:8000/  （mandalart/ ・ sticky/ ・ tree/）
```

広聴AI で作った自分のレポートで描くときは、出力ディレクトリの `public_result.json` を使う。付箋ズームと階層リストは `?data=<public_result.json の URL>` でそのまま読める。マンダラートは 3 に渡して木を作り、`?data=<木の json>` で読む（`--embeddings` を省くと 2 次元配置で割る）。

## 意見マンダラートの作り

- 広聴AI の既定の階層づくり（最細の段を k-means で切り、Ward で上へ併合する）では、親 1 つあたりの子の数が揃わない（このデータでは上位 20 クラスタの子の数が 1〜38）。マンダラートは中心＋8 マスなので、上から順に k-means（k=8）で割り、子が 8 以下になった節は意見そのものをマスに並べる。深さは枝ごとに違う（このデータで 5,848 節・最大 7 段）
- 埋め込みは長さ 1 に正規化してから割る（cosine）。乱数の種は 42
- 表札は LLM を使わず、各まとまりの中心に最も近い意見をそのまま使う。上の段ほど「〜が不足している」のような一般的な文になりやすいので、上の数段だけ LLM で表札を付ける余地がある
- マスの文字は、全文が収まる最大の大きさにする（10〜40px、ただし 1 行に 3 文字は入る大きさまで）。全マスを同じ歩みで二分探索するので、1 歩ごとの配置計算は 1 回ですむ。10px でも入らないマスだけ入る行数で「…」にする。画面の大きさが変わったら測り直す
  - 確かめた値（2026-11-08、Chromium）：幅 1280px で 3×3 は 26〜38px（中央値 31px）、9×9 は 13〜24px（中央値 18px）でどのマスも全文が入る。英語の 9×9 は 10〜24px で 81 マス中 4 マスが「…」。幅 390px では 3×3 が 12.5〜19px で全文、9×9 は 10px で 81 マス中 54 マスが「…」
  - 9×9 の盤は最小幅 620px で、狭い画面では横にスクロールする（以前は幅 390px でマスどうしが重なっていた）

## 付箋ズームの作り

最初の版は、画面を付箋 1 枚ぶんの升目に区切って升目ごとに 1 枚を毎フレーム選び直していたため、拡大すると付箋が入れ替わって震えた。今の版は地図アプリの地名ラベルと同じ考え方で、表示する前に全部決めておく。

- 付箋は画面上で一定の大きさ（拡大しても文字が読める）。位置は世界座標で固定
- 位置は、オリジナル（`canvas_kozaneba_prototype`）と同じく格子に寄せ、同じマスが埋まっていたら螺旋状に一番近い空きマスへずらす。こうすると最も寄った状態で全意見が重ならずに並ぶ（平均のずれは 0.76 マス）
- 拡大率を 2 倍ごとに 8 段に区切り、引いた側から順に、優先度の高い意見から「既に出ている付箋と重ならないなら出す」を決める。一度出た付箋は、寄った段でも必ず出ている（寄るほど付箋どうしの間隔は広がるので）
- 優先度：上位クラスタの代表（中心に最も近い意見の位置に、上位の表札を出す）→ 下位クラスタの代表（同じく下位の表札）→ 残りを下位クラスタごとに中心に近い順で、全クラスタを順に回す
- 表示側は意見ごとの「出始める拡大率」と今の拡大率を比べるだけ。出ていない意見は点で描く。出始めは 1.5 段ぶんかけて濃くする
- 付箋の文字は、全文が収まる最大の大きさにする（10〜28px）。付箋の大きさは一定なので、大きさと改行位置は付箋ごとに 1 回だけ決めて使い回す（寄っても文字は変わらない）。文字幅は 1 文字ずつ 1 回だけ測って大きさに比例させるので、大きさを何度試しても重くならない。句読点・閉じ括弧は行頭に置かない
- まとまりの代表の付箋は表札だけを、濃い色・太字で書く。代表の意見の本文は押すと「このまとまりの代表の意見」として出る。以前は表札と本文を 1 枚に入れていて、140×76px では両方の全文を入れると 10〜12px にしかならなかった
  - 確かめた値（2026-11-08、Chromium、幅 1280px で 3 段寄った状態の 134 枚）：表札の付箋 44 枚は 13〜18px、意見の付箋 90 枚は 10〜21px（大半が 13〜17px）。10px でも入らず「…」になったのは 1 枚

このデータでは、全体が 1000px に収まる拡大率で 27 枚、2 倍で 75 枚、4 倍で 228 枚、8 倍で 601 枚、16 倍で 2,058 枚が付箋になり、約 60 倍で全 21,348 件が付箋になる。

## 本体に入れる判断の目安

- 実際に使われた例がある（会議・発表・自治体での利用など）
- `public_result.json` だけで描ける（足りない項目があれば、その項目を公開契約に足す提案とセットにする）
- 本体側で保守する人がいる

## ライセンス

- コード：MIT（[LICENSE](LICENSE)）
- データ（`docs/data/` と `docs/mandalart/team-mirai-2025.json`）：CC BY 4.0。出典は上記のとおり
