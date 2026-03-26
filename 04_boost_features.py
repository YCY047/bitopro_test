"""
BitoGuard - 特徵強化版（針對 Precision 提升）
================================================
問題診斷：
  - 現有 baseline Test F1 = 0.1830
  - 瓶頸在 Precision 只有 12%，Recall 已有 44%
  - 代表模型「亂槍打鳥」，需要更精準的黑名單信號

本腳本新增三類特徵：
  A. 黑名單網路：量化接觸程度（不只是 0/1）
  B. 資金流向異常：入金→出金的速度與比例
  C. 多條件組合：同時滿足多個可疑條件的罕見程度

執行方式：
  python 04_boost_features.py
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
print("BitoGuard 特徵強化版 — 針對 Precision 提升")
print("Baseline: OOF F1=0.1795 / Test F1=0.1830")
print("=" * 60)

df = pd.read_parquet(f"{FEAT_DIR}/feature_matrix.parquet")
d = df.copy()

# ══════════════════════════════════════════════════════════
# 沿用 Baseline 的特徵（已知有效）
# ══════════════════════════════════════════════════════════
d['swap_active']            = (d['swap_tx_count'] > 0).astype(int)
d['swap_log_total']         = np.log1p(d['swap_total_twd'])
d['swap_log_max']           = np.log1p(d['swap_max_twd'])
d['swap_log_count']         = np.log1p(d['swap_tx_count'])
d['swap_amount_per_tx']     = d['swap_total_twd'] / (d['swap_tx_count'] + 1)
d['swap_high_total']        = (d['swap_total_twd'] > 50000).astype(int)
d['swap_very_high_total']   = (d['swap_total_twd'] > 200000).astype(int)
d['swap_concentration']     = d['swap_max_twd'] / (d['swap_total_twd'] + 1)
d['swap_sell_heavy']        = (d['swap_sell_ratio'] > 0.5).astype(int)

d['crypto_active']          = (d['crypto_tx_count'] > 0).astype(int)
d['crypto_log_count']       = np.log1p(d['crypto_tx_count'])
d['crypto_log_withdraw']    = np.log1p(d['crypto_withdraw_count'])
d['crypto_heavy_user']      = (d['crypto_tx_count'] > 10).astype(int)
d['crypto_withdraw_heavy']  = (d['crypto_withdraw_ratio'] > 0.7).astype(int)

d['twd_active']             = (d['twd_tx_count'] > 0).astype(int)
d['twd_withdraw_heavy']     = (d['twd_withdraw_ratio'] > 0.8).astype(int)
d['twd_quick_out']          = d['twd_is_quick_out']
d['twd_log_amount']         = np.log1p(d.get('twd_total_amount', pd.Series(0, index=d.index)))

d['blacklist_contact']      = ((d['has_blacklist_contact_1hop'] +
                                d['has_blacklist_contact_2hop']) > 0).astype(int)
d['blacklist_exposure']     = (d['has_blacklist_contact_1hop'] * 5 +
                                d['has_blacklist_contact_2hop'] * 2)

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
d['high_risk_career']       = d.get('is_high_risk_career', pd.Series(0, index=d.index))

# ══════════════════════════════════════════════════════════
# A. 黑名單網路：量化接觸程度
# ══════════════════════════════════════════════════════════
# 原本只有 0/1（有沒有接觸），現在加入程度
d['blacklist_1hop_only']    = (d['has_blacklist_contact_1hop'] == 1) & (d['has_blacklist_contact_2hop'] == 0)
d['blacklist_1hop_only']    = d['blacklist_1hop_only'].astype(int)
d['blacklist_both_hops']    = (d['has_blacklist_contact_1hop'] == 1) & (d['has_blacklist_contact_2hop'] == 1)
d['blacklist_both_hops']    = d['blacklist_both_hops'].astype(int)
# 有黑名單聯繫 + 大量 swap = 最高風險組合
d['blacklist_x_swap_heavy'] = d['blacklist_exposure'] * d['swap_log_total']
# 有黑名單聯繫 + 快速出金 = 洗錢模式
d['blacklist_x_quick_out']  = d['blacklist_contact'] * d['twd_is_quick_out']
# 有黑名單聯繫 + 高出金比率
d['blacklist_x_withdraw']   = d['blacklist_contact'] * d['twd_withdraw_ratio']

# ══════════════════════════════════════════════════════════
# B. 資金流向異常
# ══════════════════════════════════════════════════════════
# swap 後立刻出金：swap 金額占出金金額的比例
twd_total = d.get('twd_total_amount', pd.Series(1, index=d.index)).clip(lower=1)
d['swap_to_twd_ratio']      = d['swap_total_twd'] / (twd_total + 1)
# 極端快進快出：有 twd 但停留時間極短
d['extreme_quick_out']      = d['twd_is_quick_out'] * (d['twd_withdraw_ratio'] > 0.9).astype(int)
# 大額 swap + 高出金比率（資金快速流出）
d['big_swap_x_withdraw']    = d['swap_very_high_total'] * d['twd_withdraw_heavy']
# crypto 出金比率 × swap 活躍（虛實幣一起動）
d['crypto_swap_withdraw']   = d['crypto_withdraw_ratio'] * d['swap_active']
# 只用 swap 不用 twd 出金（換幣後直接走）
d['swap_no_twd']            = d['swap_active'] * (1 - d['twd_active'])
# swap 賣出 + crypto 出金（雙向變現）
d['dual_cash_out']          = d['swap_sell_heavy'] * d['crypto_withdraw_heavy']

# ══════════════════════════════════════════════════════════
# C. 多條件組合的罕見程度（提升 Precision 的核心）
# ══════════════════════════════════════════════════════════
# 可疑行為計數：同時滿足越多條件，越可疑
d['suspicious_count'] = (
    d['swap_high_total'] +           # 大額 swap
    d['twd_withdraw_heavy'] +        # 高出金比率
    d['twd_is_quick_out'] +          # 快速出金
    d['crypto_withdraw_heavy'] +     # crypto 高出金
    d['blacklist_contact'] +         # 黑名單聯繫
    d['swap_sell_heavy'] +           # swap 以賣出為主
    (d['twd_withdraw_ratio'] > 0.9).astype(int) +   # 極高出金比率
    (d['swap_total_twd'] > 100000).astype(int) +    # swap 超大額
    d['multi_channel'].clip(upper=1)                 # 多管道
)
# 高度可疑用戶（同時滿足 3 個以上條件）
d['highly_suspicious']      = (d['suspicious_count'] >= 3).astype(int)
d['extremely_suspicious']   = (d['suspicious_count'] >= 5).astype(int)

# swap 大額 + 多管道 + 快速出金（三重信號）
d['triple_signal']          = (
    d['swap_high_total'] *
    (d['multi_channel'] >= 2).astype(int) *
    d['twd_is_quick_out']
)
# 黑名單 + 任何一個財務異常（高確信度）
d['blacklist_plus_financial'] = d['blacklist_contact'] * (
    d['swap_high_total'] + d['twd_withdraw_heavy'] + d['twd_is_quick_out']
).clip(upper=1)

# ══════════════════════════════════════════════════════════
# D. 對數比率特徵（讓分佈更對稱，有助於模型學習）
# ══════════════════════════════════════════════════════════
d['log_swap_per_crypto']    = np.log1p(d['swap_total_twd']) - np.log1p(d['crypto_tx_count'])
d['log_withdraw_intensity'] = np.log1p(d['twd_withdraw_ratio'] * d['swap_total_twd'])

# ══════════════════════════════════════════════════════════
# 印出新特徵的區分力
# ══════════════════════════════════════════════════════════
NEW_FEATURES = [
    'blacklist_1hop_only', 'blacklist_both_hops', 'blacklist_x_swap_heavy',
    'blacklist_x_quick_out', 'blacklist_x_withdraw',
    'swap_to_twd_ratio', 'extreme_quick_out', 'big_swap_x_withdraw',
    'crypto_swap_withdraw', 'swap_no_twd', 'dual_cash_out',
    'suspicious_count', 'highly_suspicious', 'extremely_suspicious',
    'triple_signal', 'blacklist_plus_financial',
    'log_swap_per_crypto', 'log_withdraw_intensity',
]

bl = d[d['status'] == 1]
nm = d[d['status'] == 0]
print("\n新特徵區分力（黑名單均值 / 正常均值）：")
print(f"  {'特徵名稱':<35} {'黑名單':>8}  {'正常':>8}  {'倍數':>6}")
print("  " + "-" * 62)
ratios = {}
for c in NEW_FEATURES:
    b = bl[c].mean()
    n = nm[c].mean()
    ratio = b / (n + 1e-9)
    ratios[c] = ratio
    bar = "█" * min(int(ratio), 20)
    print(f"  {c:<35} {b:>8.4f}  {n:>8.4f}  {ratio:>6.2f}x  {bar}")

# ══════════════════════════════════════════════════════════
# 建立最終特徵矩陣
# ══════════════════════════════════════════════════════════
EXCLUDE = [
    'trade_night_ratio', 'trade_buy_ratio', 'trade_tx_count',
    'trade_unique_ip', 'crypto_night_ratio',
    'has_blacklist_contact_1hop', 'has_blacklist_contact_2hop',
]
FEAT_COLS = [c for c in d.columns if c not in ['user_id', 'status'] + EXCLUDE]
print(f"\n總特徵數: {len(FEAT_COLS)}（Baseline: 57）")

X = d[FEAT_COLS].fillna(0).values.astype(float)
y = d['status'].values.astype(int)

# ══════════════════════════════════════════════════════════
# 乾淨的 CV 框架（與 Baseline 完全相同，只換特徵）
# ══════════════════════════════════════════════════════════
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

skf       = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
smote     = SMOTE(random_state=42, k_neighbors=3)
scale_pos = int((y_train == 0).sum() / max((y_train == 1).sum(), 1))
oof_probs = np.zeros(len(X_train))

print("\n[1] 5-Fold CV...")
for fold, (tr_idx, val_idx) in enumerate(skf.split(X_train, y_train), 1):
    X_tr, X_val = X_train[tr_idx], X_train[val_idx]
    y_tr, y_val = y_train[tr_idx], y_train[val_idx]

    scaler  = StandardScaler()
    X_tr_s  = scaler.fit_transform(X_tr)
    X_val_s = scaler.transform(X_val)

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

    prob = (xgb.predict_proba(X_val_s)[:, 1] * 0.6 +
            lgbm.predict_proba(X_val_s)[:, 1] * 0.4)
    oof_probs[val_idx] = prob

    prec, rec, thrs = precision_recall_curve(y_val, prob)
    f1s = 2 * prec * rec / (prec + rec + 1e-9)
    print(f"  Fold {fold}: OOF F1 = {np.max(f1s):.4f}")

prec_oof, rec_oof, thrs_oof = precision_recall_curve(y_train, oof_probs)
f1s_oof  = 2 * prec_oof * rec_oof / (prec_oof + rec_oof + 1e-9)
best_idx = np.argmax(f1s_oof[:-1])
best_thr = float(thrs_oof[best_idx])
oof_f1   = float(f1s_oof[best_idx])

print(f"\n✅ OOF F1 = {oof_f1:.4f}  （Baseline: 0.1795）")
delta_oof = oof_f1 - 0.1795
print(f"   變化量  = {delta_oof:+.4f}  {'⬆ 有改善！' if delta_oof > 0.002 else '⬇ 沒有改善' if delta_oof < -0.002 else '→ 持平'}")

# 最終模型
print("\n[2] 訓練最終模型...")
final_scaler = StandardScaler()
X_train_s = final_scaler.fit_transform(X_train)
X_test_s  = final_scaler.transform(X_test)

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

y_prob_test = (final_xgb.predict_proba(X_test_s)[:, 1] * 0.6 +
               final_lgbm.predict_proba(X_test_s)[:, 1] * 0.4)
y_pred_test = (y_prob_test >= best_thr).astype(int)
test_f1 = f1_score(y_test, y_pred_test)

print(f"\n{'='*60}")
print("對比 Baseline")
print(f"{'='*60}")
print(f"  {'指標':<20} {'Baseline':>10}  {'本版本':>10}  {'變化':>8}")
print(f"  {'-'*52}")
print(f"  {'OOF F1':<20} {'0.1795':>10}  {oof_f1:>10.4f}  {oof_f1-0.1795:>+8.4f}")
print(f"  {'Test F1':<20} {'0.1830':>10}  {test_f1:>10.4f}  {test_f1-0.1830:>+8.4f}")
print(f"  {'Test Precision':<20} {'0.1158':>10}  {precision_score(y_test, y_pred_test, zero_division=0):>10.4f}")
print(f"  {'Test Recall':<20} {'0.4360':>10}  {recall_score(y_test, y_pred_test, zero_division=0):>10.4f}")
print(f"{'='*60}")
print()
print(classification_report(y_test, y_pred_test,
                             target_names=["正常", "黑名單"], digits=4))

# 儲存
with open(f"{OUTPUT_DIR}/boost_features_model.pkl", "wb") as f:
    pickle.dump({
        "xgb": final_xgb, "lgbm": final_lgbm,
        "scaler": final_scaler, "threshold": best_thr,
        "features": FEAT_COLS, "oof_f1": oof_f1, "test_f1": test_f1,
    }, f)

print("模型已儲存: ./data/model_output/boost_features_model.pkl")
print("\n把 OOF F1 和 Test F1 告訴我，我們決定下一步方向")