# Debugging log / デバッグ記録

Planned entries (filled in Phase 2) / Phase 2 で記録予定:

| Bug / バグ | Symptom to measure / 測定する症状 | Status |
|---|---|---|
| RGB/BGR swap / RGB・BGR取り違え | accuracy drop vs. baseline | todo |
| Mask resized with linear interpolation / マスクの線形補間 | fractional labels, boundary IoU drop | todo |
| Double normalization / 正規化の二重がけ | loss plateau, accuracy drop | todo |

Safeguard already in place / 既存の安全策: `tests/test_overfit.py` (tiny-batch overfit check).
