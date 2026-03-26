"""
BitoGuard - 乾淨 Baseline（無資料洩漏）
================================================
修正三個核心問題：
  1. Scaler 移入 CV loop 內，驗證折只 transform
  2. 最終報告用 OOF 預測，不用全量 in-sample
  3. Hold-out test set 全程隔離，最後才碰

執行方式：
  pip install xgboost lightgbm imbalanced-learn scikit-learn pandas pyarrow
  python 04_clean_baseline.py
"""

import pandas as pd
import numpy as np
import os, pickle, warnings
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import (f1_score, precision_score, recall_score,
                              classification_report, precision_recall_curve)
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

warnings.filterwarnings("ignore")

FEAT_DIR   = "./data/features"
OUTPUT_DIR = "./data/model_output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 60)
print("BitoGuard 乾淨 Baseline — 無資料洩漏版")
print("=" * 60)

# ── 1. 載入資料 ───────────────────────────────────────────
df = pd.read_parquet(f"{FEAT_DIR}/feature_matrix.parquet")
n_black = (df['status'] == 1).sum()
n_total = len(df)
print(f"總用戶: {n_total:,}  黑名單: {n_black:,} ({n_black/n_total*100:.2f}%)")

# ── 2. 特徵工程 ───────────────────────────────────────────
# 注意：這些計算只用到當前行，不涉及跨行統計，沒有洩漏問題
d = df.copy()

# swap 特徵（最強信號：黑名單 49.9% vs 正常 22.4%）
d['swap_active']            = (d['swap_tx_count'] > 0).astype(int)
d['swap_log_total']         = np.log1p(d['swap_total_twd'])
d['swap_log_max']           = np.log1p(d['swap_max_twd'])
d['swap_log_count']         = np.log1p(d['swap_tx_count'])
d['swap_amount_per_tx']     = d['swap_total_twd'] / (d['swap_tx_count'] + 1)
d['swap_high_total']        = (d['swap_total_twd'] > 50000).astype(int)
d['swap_very_high_total']   = (d['swap_total_twd'] > 200000).astype(int)
d['swap_concentration']     = d['swap_max_twd'] / (d['swap_total_twd'] + 1)  # 集中在單筆？
d['swap_sell_heavy']        = (d['swap_sell_ratio'] > 0.5).astype(int)

# crypto 特徵
d['crypto_active']          = (d['crypto_tx_count'] > 0).astype(int)
d['crypto_log_count']       = np.log1p(d['crypto_tx_count'])
d['crypto_log_withdraw']    = np.log1p(d['crypto_withdraw_count'])
d['crypto_heavy_user']      = (d['crypto_tx_count'] > 10).astype(int)
d['crypto_withdraw_heavy']  = (d['crypto_withdraw_ratio'] > 0.7).astype(int)

# twd 特徵
d['twd_active']             = (d['twd_tx_count'] > 0).astype(int)
d['twd_withdraw_heavy']     = (d['twd_withdraw_ratio'] > 0.8).astype(int)
d['twd_quick_out']          = d['twd_is_quick_out']
d['twd_log_amount']         = np.log1p(d.get('twd_total_amount', pd.Series(0, index=d.index)))

# 黑名單關聯（有就很強，但只有 5.4% 用戶有）
d['blacklist_contact']      = ((d['has_blacklist_contact_1hop'] +
                                d['has_blacklist_contact_2hop']) > 0).astype(int)
d['blacklist_exposure']     = (d['has_blacklist_contact_1hop'] * 5 +
                                d['has_blacklist_contact_2hop'] * 2)

# 跨管道組合特徵（同時使用多種管道才可疑）
d['multi_channel']          = (
    (d['swap_tx_count'] > 0).astype(int) +
    (d['crypto_tx_count'] > 0).astype(int) +
    (d['twd_tx_count'] > 0).astype(int)
)
d['swap_AND_blacklist']     = d['swap_active'] * d['blacklist_contact']
d['swap_x_withdraw']        = d['swap_log_total'] * d['twd_withdraw_ratio']
d['swap_x_crypto']          = d['swap_log_total'] * d['crypto_log_count']
d['crypto_withdraw_x_swap'] = d['crypto_withdraw_ratio'] * d['swap_log_total']
d['high_swap_x_withdraw']   = d['swap_high_total'] * d['twd_withdraw_heavy']

# KYC
d['high_risk_career']       = d.get('is_high_risk_career', pd.Series(0, index=d.index))

# 排除在真實資料中方向錯誤的特徵（diag.py 顯示黑名單反而比正常低）
EXCLUDE = [
    'trade_night_ratio',
    'trade_buy_ratio',
    'trade_tx_count',
    'trade_unique_ip',
    'crypto_night_ratio',
    'has_blacklist_contact_1hop',   # 已合併進 blacklist_contact / blacklist_exposure
    'has_blacklist_contact_2hop',
]

FEAT_COLS = [c for c in d.columns if c not in ['user_id', 'status'] + EXCLUDE]
X = d[FEAT_COLS].fillna(0).values.astype(float)
y = d['status'].values.astype(int)

print(f"特徵數: {len(FEAT_COLS)}")

# ── 3. 切出 hold-out test set（全程不能碰）────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)
print(f"\nTrain: {len(X_train):,}  Test: {len(X_test):,}")
print(f"Train 黑名單: {y_train.sum():,}  Test 黑名單: {y_test.sum():,}")

# ── 4. 5-Fold CV，收集 OOF 預測 ──────────────────────────
print("\n[1] 5-Fold CV（收集 OOF 預測，這才是真實分數）...")

skf   = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
smote = SMOTE(random_state=42, k_neighbors=3)
scale_pos = int((y_train == 0).sum() / max((y_train == 1).sum(), 1))
print(f"scale_pos_weight: {scale_pos}")

oof_probs = np.zeros(len(X_train))

for fold, (tr_idx, val_idx) in enumerate(skf.split(X_train, y_train), 1):
    X_tr, X_val = X_train[tr_idx], X_train[val_idx]
    y_tr, y_val = y_train[tr_idx], y_train[val_idx]

    # ✅ 關鍵：Scaler 只 fit 訓練折，驗證折只 transform
    scaler  = StandardScaler()
    X_tr_s  = scaler.fit_transform(X_tr)
    X_val_s = scaler.transform(X_val)

    # ✅ SMOTE 在 scale 後的訓練折上做（不污染驗證折）
    try:
        X_res, y_res = smote.fit_resample(X_tr_s, y_tr)
    except Exception:
        X_res, y_res = X_tr_s, y_tr

    xgb = XGBClassifier(
        n_estimators=500, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        scale_pos_weight=scale_pos,
        random_state=42, eval_metric="logloss", verbosity=0,
    )
    lgbm = LGBMClassifier(
        n_estimators=500, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        scale_pos_weight=scale_pos,
        random_state=42, verbose=-1,
    )
    xgb.fit(X_res, y_res, eval_set=[(X_val_s, y_val)], verbose=False)
    lgbm.fit(X_res, y_res)

    # ✅ 只預測沒看過的驗證折
    prob = (xgb.predict_proba(X_val_s)[:, 1] * 0.6 +
            lgbm.predict_proba(X_val_s)[:, 1] * 0.4)
    oof_probs[val_idx] = prob

    # 這個 fold 的最佳 F1（僅供參考）
    prec, rec, thrs = precision_recall_curve(y_val, prob)
    f1s = 2 * prec * rec / (prec + rec + 1e-9)
    print(f"  Fold {fold}: OOF F1 = {np.max(f1s):.4f}")

# ✅ 用彙整的 OOF 找最佳閾值（這才是真實的 CV 表現）
prec_oof, rec_oof, thrs_oof = precision_recall_curve(y_train, oof_probs)
f1s_oof  = 2 * prec_oof * rec_oof / (prec_oof + rec_oof + 1e-9)
best_idx  = np.argmax(f1s_oof[:-1])
best_thr  = float(thrs_oof[best_idx])
oof_f1    = float(f1s_oof[best_idx])
oof_prec  = float(prec_oof[best_idx])
oof_rec   = float(rec_oof[best_idx])

print(f"\n{'='*60}")
print(f"✅ 真實 OOF F1        = {oof_f1:.4f}")
print(f"   OOF Precision     = {oof_prec:.4f}")
print(f"   OOF Recall        = {oof_rec:.4f}")
print(f"   最佳閾值           = {best_thr:.4f}")
print(f"{'='*60}")
print("（這個數字才是可信的，下面的 test set 才是最終驗證）")

# ── 5. 訓練最終模型（用全部 train set）───────────────────
print("\n[2] 訓練最終模型...")
final_scaler = StandardScaler()
X_train_s    = final_scaler.fit_transform(X_train)
X_test_s     = final_scaler.transform(X_test)   # ✅ test 只 transform

try:
    X_res, y_res = smote.fit_resample(X_train_s, y_train)
except Exception:
    X_res, y_res = X_train_s, y_train

final_xgb = XGBClassifier(
    n_estimators=500, max_depth=6, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8,
    scale_pos_weight=scale_pos,
    random_state=42, eval_metric="logloss", verbosity=0,
)
final_lgbm = LGBMClassifier(
    n_estimators=500, max_depth=6, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8,
    scale_pos_weight=scale_pos,
    random_state=42, verbose=-1,
)
final_xgb.fit(X_res, y_res, verbose=False)
final_lgbm.fit(X_res, y_res)

# ── 6. Test set 評估（最終真實分數）──────────────────────
y_prob_test = (final_xgb.predict_proba(X_test_s)[:, 1] * 0.6 +
               final_lgbm.predict_proba(X_test_s)[:, 1] * 0.4)
y_pred_test = (y_prob_test >= best_thr).astype(int)

test_f1   = f1_score(y_test, y_pred_test)
test_prec = precision_score(y_test, y_pred_test, zero_division=0)
test_rec  = recall_score(y_test, y_pred_test, zero_division=0)

print(f"\n{'='*60}")
print("最終結果摘要")
print(f"{'='*60}")
print(f"  OOF F1  （CV 估計，訓練中可見）: {oof_f1:.4f}")
print(f"  Test F1 （真實，全程隔離）     : {test_f1:.4f}")
print(f"  Test Precision                : {test_prec:.4f}")
print(f"  Test Recall                   : {test_rec:.4f}")
print(f"{'='*60}")
print()
print(classification_report(y_test, y_pred_test,
                             target_names=["正常", "黑名單"], digits=4))

# ── 7. 閾值敏感度（幫助了解 Precision/Recall 權衡）────────
print("閾值敏感度分析（Test Set）：")
print(f"  {'閾值':>5}  {'Precision':>10}  {'Recall':>8}  {'F1':>8}  {'預測黑名單數':>12}")
for thr in np.arange(0.05, 0.96, 0.05):
    yp  = (y_prob_test >= thr).astype(int)
    p_  = precision_score(y_test, yp, zero_division=0)
    r_  = recall_score(y_test, yp, zero_division=0)
    f_  = f1_score(y_test, yp, zero_division=0)
    n_  = int(yp.sum())
    mark = " ← 目前選擇" if abs(thr - best_thr) < 0.025 else ""
    print(f"  {thr:.2f}  {p_:>10.4f}  {r_:>8.4f}  {f_:>8.4f}  {n_:>12}{mark}")

# ── 8. 儲存模型 ───────────────────────────────────────────
model_path = f"{OUTPUT_DIR}/clean_baseline_model.pkl"
with open(model_path, "wb") as f:
    pickle.dump({
        "xgb":       final_xgb,
        "lgbm":      final_lgbm,
        "scaler":    final_scaler,
        "threshold": best_thr,
        "features":  FEAT_COLS,
        "oof_f1":    oof_f1,
        "test_f1":   test_f1,
    }, f)
print(f"\n模型已儲存: {model_path}")

print("\n" + "="*60)
print("下一步：把 OOF F1 和 Test F1 的數字告訴我")
print("我們才能決定下一個優化方向")
print("="*60)