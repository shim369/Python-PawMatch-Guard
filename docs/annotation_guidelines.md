# Annotation Guidelines / アノテーションガイドライン

Version 1.0 · Tool: Label Studio · Format: axis-aligned boxes `[x1, y1, x2, y2]` in pixels.
バージョン 1.0 · ツール: Label Studio · 形式: ピクセル座標の矩形 `[x1, y1, x2, y2]`

## Classes / クラス

| ID | Class | What to box / 対象 |
|---|---|---|
| 0 | `animal` | The whole visible cat or dog (head to tail). 見えている犬猫の全体 |
| 1 | `face` | A human face (any size where eyes or mouth are recognizable). 目か口が判別できる人の顔 |
| 2 | `plate` | A vehicle license plate. 車のナンバープレート |
| 3 | `text` | A line or block of readable text (signs, name tags, documents). 読める文字の行・まとまり |

Class IDs are fixed; `check-annotations` rejects IDs outside 0–3.
クラスIDは固定です。0〜3以外はチェックで弾かれます。

> **Note / 注意:** the public Oxford-IIIT Pet boxes used to bootstrap the detector cover the animal's *head* only, so the current `animal_head` detector differs from the whole-animal `animal` class defined here. 公開データの矩形は頭部のみで、現在の検出器は本ガイドラインの全身 `animal` とは異なります。

## Box rules / 矩形の付け方

1. **Tight box.** Touch the outermost visible pixel of the object; do not add margin.
   対象の外側ぎりぎりに付ける。余白は入れない。
2. **Partially visible objects:** box only the visible part if at least ~30% is visible and it
   is still recognizable; skip anything smaller. 一部だけ写っている場合は、見えている部分のみを、
   約30%以上見えて判別できるときに付ける。それ未満は付けない。
3. **Occlusion:** box the visible extent, even if another object overlaps it. 重なりは見える範囲で付ける。
4. **Groups:** one box per individual animal / face. Do not draw a box around a crowd.
   1匹・1人につき1つ。群れ全体を1つで囲まない。
5. **Text:** one box per line when lines are separated; one box for a tightly packed block.
   行が離れていれば行ごと、密なブロックはまとめて1つ。
6. **Boxes must stay inside the image and have non-zero area.** 画像外・面積ゼロは不可。
7. **Illustrations, toys, photos of photos:** do not label. イラスト・ぬいぐるみ・写真の写真は付けない。

## Hard cases and answers / 迷う例とその答え

| Situation / 状況 | Answer / 答え |
|---|---|
| A cat's tail leaves the frame | Box up to the frame edge (clip to image). 枠の端まで(画像内にクリップ) |
| Only a dog's ears and eyes peek over a wall | Label as `animal` if the head is recognizable. 頭が判別できれば付ける |
| Face in a mirror or on a screen | Label it if clearly recognizable. はっきり判別できれば付ける |
| Plate blurred by motion, unreadable | Still label `plate` (the shape is the privacy risk). 読めなくても付ける |
| Cat and dog cuddling, overlapping | Two `animal` boxes, overlap allowed. 重なっても2つ付ける |
| Tiny text far in the background (< ~8 px high) | Skip. 小さすぎる文字は付けない |
| Face covered by a mask or sunglasses | Label `face` if the face region is evident. 顔の範囲が分かれば付ける |
| Doubt whether an object is a face/plate/text | Label it. Missing personal data is worse than a false box. 迷ったら付ける(見逃しが重い) |

## Quality procedure / 品質管理の手順

1. **Consistency checks (automatic):** out-of-bounds coordinates, zero area, class ID out of
   range, mask/box mismatch → `src/quality/annotation_checks.py`.
2. **Repeat annotation:** re-annotate ~10% of images after at least one day without looking at
   the first pass; measure agreement with box IoU (`src/quality/agreement.py`). Target: mean
   IoU ≥ 0.80 and match rate ≥ 0.90. If lower, revise this document.
3. **Noisy-label scan:** train, then flag high-loss / disagreeing samples for manual review
   (`pawmatch detect-noise`).

1. 自動チェック: 画像外の座標、面積ゼロ、クラスID範囲外、マスクと矩形の不整合。
2. 再アノテーション: 約10%を1日以上空けて再作業し、矩形IoUで一致度を測る。目標は平均IoU 0.80以上、対応率0.90以上。下回ればこの文書を改訂する。
3. ノイズラベル走査: 学習後、損失が高い・予測と食い違うサンプルを目視確認に回す。
