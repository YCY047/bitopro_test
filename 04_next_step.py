"""
BitoGuard - 下一步優化：兩個方向
================================================
方向 A：調整 scale_pos_weight + sampling 策略
  - 問題：Recall=41% 其實太高，但 Precision 才 12%
  - 降低 scale_pos_weight 讓模型更保守，提高 Precision
  - 同時測試 SMOTE 不同 k_neighbors

方向 B：針對「無黑名單聯繫」的黑名單用戶強化特徵
  - 94.6% 的黑名單沒有黑名單聯繫
  - 需要更細緻的 swap + twd 行為特徵來抓這群人

本腳本同時跑兩個方向，比較結果
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
print("BitoGuard — 下一步優化")
print("Previous best: OOF F1=0.1804 / Test F1=0.1907")
print("=" * 60)

df = pd.read_parquet(f"{FEAT_DIR}/feature_matrix.parquet")
d = df.copy()

# ── 完整特徵工程（沿用 boost_features 版本）─────────────
d['swap_active']              = (d['swap_tx_count'] > 0).astype(int)
d['swap_log_total']           = np.log1p(d['swap_total_twd'])
d['swap_log_max']             = np.log1p(d['swap_max_twd'])
d['swap_log_count']           = np.log1p(d['swap_tx_count'])
d['swap_amount_per_tx']       = d['swap_total_twd'] / (d['swap_tx_count'] + 1)
d['swap_high_total']          = (d['swap_total_twd'] > 50000).astype(int)
d['swap_very_high_total']     = (d['swap_total_twd'] > 200000).astype(int)
d['swap_concentration']       = d['swap_max_twd'] / (d['swap_total_twd'] + 1)
d['swap_sell_heavy']          = (d['swap_sell_ratio'] > 0.5).astype(int)

d['crypto_active']            = (d['crypto_tx_count'] > 0).astype(int)
d['crypto_log_count']         = np.log1p(d['crypto_tx_count'])
d['crypto_log_withdraw']      = np.log1p(d['crypto_withdraw_count'])
d['crypto_heavy_user']        = (d['crypto_tx_count'] > 10).astype(int)
d['crypto_withdraw_heavy']    = (d['crypto_withdraw_ratio'] > 0.7).astype(int)

d['twd_active']               = (d['twd_tx_count'] > 0).astype(int)
d['twd_withdraw_heavy']       = (d['twd_withdraw_ratio'] > 0.8).astype(int)
d['twd_quick_out']            = d['twd_is_quick_out']
d['twd_log_amount']           = np.log1p(d.get('twd_total_amount', pd.Series(0, index=d.index)))

d['blacklist_contact']        = ((d['has_blacklist_contact_1hop'] +
                                  d['has_blacklist_contact_2hop']) > 0).astype(int)
d['blacklist_exposure']       = (d['has_blacklist_contact_1hop'] * 5 +
                                  d['has_blacklist_contact_2hop'] * 2)
d['blacklist_both_hops']      = ((d['has_blacklist_contact_1hop'] == 1) &
                                  (d['has_blacklist_contact_2hop'] == 1)).astype(int)
d['blacklist_x_swap_heavy']   = d['blacklist_exposure'] * d['swap_log_total']
d['blacklist_x_quick_out']    = d['blacklist_contact'] * d['twd_is_quick_out']
d['blacklist_x_withdraw']     = d['blacklist_contact'] * d['twd_withdraw_ratio']
d['blacklist_plus_financial']  = d['blacklist_contact'] * (
    d['swap_high_total'] + d['twd_withdraw_heavy'] + d['twd_is_quick_out']
).clip(upper=1)

d['multi_channel']            = (
    (d['swap_tx_count'] > 0).astype(int) +
    (d['crypto_tx_count'] > 0).astype(int) +
    (d['twd_tx_count'] > 0).astype(int)
)
d['swap_AND_blacklist']       = d['swap_active'] * d['blacklist_contact']
d['swap_x_withdraw']          = d['swap_log_total'] * d['twd_withdraw_ratio']
d['swap_x_crypto']            = d['swap_log_total'] * d['crypto_log_count']
d['crypto_withdraw_x_swap']   = d['crypto_withdraw_ratio'] * d['swap_log_total']
d['high_swap_x_withdraw']     = d['swap_high_total'] * d['twd_withdraw_heavy']
d['high_risk_career']         = d.get('is_high_risk_career', pd.Series(0, index=d.index))
d['swap_to_twd_ratio']        = d['swap_total_twd'] / (d.get('twd_total_amount', pd.Series(1, index=d.index)).clip(lower=1) + 1)
d['crypto_swap_withdraw']     = d['crypto_withdraw_ratio'] * d['swap_active']
d['dual_cash_out']            = d['swap_sell_heavy'] * d['crypto_withdraw_heavy']
d['suspicious_count']         = (
    d['swap_high_total'] + d['twd_withdraw_heavy'] + d['twd_is_quick_out'] +
    d['crypto_withdraw_heavy'] + d['blacklist_contact'] + d['swap_sell_heavy'] +
    (d['twd_withdraw_ratio'] > 0.9).astype(int) +
    (d['swap_total_twd'] > 100000).astype(int) +
    d['multi_channel'].clip(upper=1)
)
d['highly_suspicious']        = (d['suspicious_count'] >= 3).astype(int)
d['extremely_suspicious']     = (d['suspicious_count'] >= 5).astype(int)
d['triple_signal']            = (
    d['swap_high_total'] *
    (d['multi_channel'] >= 2).astype(int) *
    d['twd_is_quick_out']
)
d['log_swap_per_crypto']      = np.log1p(d['swap_total_twd']) - np.log1p(d['crypto_tx_count'])
d['log_withdraw_intensity']   = np.log1p(d['twd_withdraw_ratio'] * d['swap_total_twd'])

# ── 方向 B 新增：針對無黑名單聯繫的黑名單用戶 ──────────
# 這群人（94.6%的黑名單）只能靠行為特徵區分
# 更細緻的 swap 門檻
d['swap_tier1']               = (d['swap_total_twd'].between(10000, 50000)).astype(int)
d['swap_tier2']               = (d['swap_total_twd'].between(50000, 200000)).astype(int)
d['swap_tier3']               = (d['swap_total_twd'] > 200000).astype(int)

# swap 次數的細緻分層
d['swap_freq_low']            = (d['swap_tx_count'].between(1, 3)).astype(int)
d['swap_freq_mid']            = (d['swap_tx_count'].between(4, 10)).astype(int)
d['swap_freq_high']           = (d['swap_tx_count'] > 10).astype(int)

# 每筆平均金額（大額少次 vs 小額多次）
d['swap_avg_amount']          = d['swap_total_twd'] / (d['swap_tx_count'] + 1)
d['swap_large_single']        = (d['swap_avg_amount'] > 30000).astype(int)

# twd 出金速度分層
d['withdraw_ratio_tier1']     = (d['twd_withdraw_ratio'].between(0.5, 0.7)).astype(int)
d['withdraw_ratio_tier2']     = (d['twd_withdraw_ratio'].between(0.7, 0.9)).astype(int)
d['withdraw_ratio_tier3']     = (d['twd_withdraw_ratio'] > 0.9).astype(int)

# crypto 出金的細緻特徵
d['crypto_withdraw_tier']     = pd.cut(
    d['crypto_withdraw_ratio'],
    bins=[-0.01, 0.3, 0.6, 0.8, 1.01],
    labels=[0, 1, 2, 3]
).astype(float)

# 全管道高活躍（swap + crypto + twd 都很活躍）
d['all_channel_active']       = (
    (d['swap_tx_count'] > 2).astype(int) *
    (d['crypto_tx_count'] > 3).astype(int) *
    (d['twd_tx_count'] > 2).astype(int)
)

# swap 大額 + crypto 出金（最典型的洗錢路徑）
d['swap_to_crypto_cashout']   = d['swap_tier2'] * d['crypto_withdraw_heavy']
d['swap_large_x_crypto_out']  = d['swap_log_total'] * d['crypto_withdraw_ratio']

# KYC 快速完成（可疑：正常用戶不會急著完成 KYC）
d['kyc_fast']                 = (d.get('kyc_l2_delay_days',
                                 pd.Series(999, index=d.index)) < 7).astype(int)

# 印出新特徵區分力
NEW_B_FEATURES = [
    'swap_tier1', 'swap_tier2', 'swap_tier3',
    'swap_freq_low', 'swap_freq_mid', 'swap_freq_high',
    'swap_large_single', 'swap_avg_amount',
    'withdraw_ratio_tier1', 'withdraw_ratio_tier2', 'withdraw_ratio_tier3',
    'all_channel_active', 'swap_to_crypto_cashout', 'swap_large_x_crypto_out',
    'kyc_fast',
]
bl = d[d['status'] == 1]
nm = d[d['status'] == 0]
print("\n方向 B 新特徵區分力：")
print(f"  {'特徵名稱':<35} {'黑名單':>8}  {'正常':>8}  {'倍數':>6}")
print("  " + "-" * 62)
for c in NEW_B_FEATURES:
    b = bl[c].mean()
    n = nm[c].mean()
    ratio = b / (n + 1e-9)
    bar = "█" * min(int(ratio * 3), 20)
    print(f"  {c:<35} {b:>8.4f}  {n:>8.4f}  {ratio:>6.2f}x  {bar}")

EXCLUDE = [
    'trade_night_ratio', 'trade_buy_ratio', 'trade_tx_count',
    'trade_unique_ip', 'crypto_night_ratio',
    'has_blacklist_contact_1hop', 'has_blacklist_contact_2hop',
]
FEAT_COLS = [c for c in d.columns if c not in ['user_id', 'status'] + EXCLUDE]
print(f"\n總特徵數: {len(FEAT_COLS)}")

X = d[FEAT_COLS].fillna(0).values.astype(float)
y = d['status'].values.astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)
scale_pos = int((y_train == 0).sum() / max((y_train == 1).sum(), 1))

# ── 方向 A：搜尋最佳 scale_pos_weight ───────────────────
print("\n" + "="*60)
print("方向 A：搜尋最佳 scale_pos_weight")
print("（降低 scale_pos_weight 讓模型更保守，提高 Precision）")
print("="*60)

def run_cv_with_spw(X_train, y_train, spw, n_splits=5):
    skf       = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    smote     = SMOTE(random_state=42, k_neighbors=3)
    oof_probs = np.zeros(len(X_train))

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
            scale_pos_weight=spw,
            random_state=42, eval_metric="logloss", verbosity=0,
        )
        lgbm = LGBMClassifier(
            n_estimators=500, max_depth=6, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            scale_pos_weight=spw,
            random_state=42, verbose=-1,
        )
        xgb.fit(X_res, y_res, eval_set=[(X_val_s, y_val)], verbose=False)
        lgbm.fit(X_res, y_res)

        prob = (xgb.predict_proba(X_val_s)[:, 1] * 0.6 +
                lgbm.predict_proba(X_val_s)[:, 1] * 0.4)
        oof_probs[val_idx] = prob

    prec, rec, thrs = precision_recall_curve(y_train, oof_probs)
    f1s = 2 * prec * rec / (prec + rec + 1e-9)
    best_idx = np.argmax(f1s[:-1])
    return float(f1s[best_idx]), float(thrs[best_idx]), oof_probs

print(f"\n  base scale_pos = {scale_pos}")
print(f"  {'SPW':>5}  {'OOF F1':>8}  {'最佳閾值':>8}")

best_spw     = scale_pos
best_oof_f1  = 0
best_thr     = 0.5
best_oof_probs = None

# 測試從 1/4 到 2 倍的 scale_pos_weight
for mult in [0.25, 0.5, 1.0, 1.5, 2.0]:
    spw = max(1, int(scale_pos * mult))
    oof_f1, thr, oof_probs = run_cv_with_spw(X_train, y_train, spw)
    mark = ""
    if oof_f1 > best_oof_f1:
        best_oof_f1    = oof_f1
        best_spw       = spw
        best_thr       = thr
        best_oof_probs = oof_probs
        mark           = " ← 最佳"
    print(f"  {spw:>5}  {oof_f1:>8.4f}  {thr:>8.4f}{mark}")

print(f"\n最佳 SPW: {best_spw}  OOF F1: {best_oof_f1:.4f}")

# ── 訓練最終模型（最佳 SPW）──────────────────────────────
print("\n訓練最終模型...")
smote_final  = SMOTE(random_state=42, k_neighbors=3)
final_scaler = StandardScaler()
X_train_s    = final_scaler.fit_transform(X_train)
X_test_s     = final_scaler.transform(X_test)

try:
    X_res, y_res = smote_final.fit_resample(X_train_s, y_train)
except Exception:
    X_res, y_res = X_train_s, y_train

final_xgb = XGBClassifier(
    n_estimators=500, max_depth=6, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8,
    scale_pos_weight=best_spw,
    random_state=42, eval_metric="logloss", verbosity=0,
)
final_lgbm = LGBMClassifier(
    n_estimators=500, max_depth=6, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8,
    scale_pos_weight=best_spw,
    random_state=42, verbose=-1,
)
final_xgb.fit(X_res, y_res, verbose=False)
final_lgbm.fit(X_res, y_res)

y_prob_test = (final_xgb.predict_proba(X_test_s)[:, 1] * 0.6 +
               final_lgbm.predict_proba(X_test_s)[:, 1] * 0.4)
y_pred_test = (y_prob_test >= best_thr).astype(int)
test_f1     = f1_score(y_test, y_pred_test)
test_prec   = precision_score(y_test, y_pred_test, zero_division=0)
test_rec    = recall_score(y_test, y_pred_test, zero_division=0)

print(f"\n{'='*60}")
print("最終對比")
print(f"{'='*60}")
print(f"  {'指標':<20} {'Baseline':>10}  {'Boost':>10}  {'本版本':>10}")
print(f"  {'-'*55}")
print(f"  {'OOF F1':<20} {'0.1795':>10}  {'0.1804':>10}  {best_oof_f1:>10.4f}")
print(f"  {'Test F1':<20} {'0.1830':>10}  {'0.1907':>10}  {test_f1:>10.4f}")
print(f"  {'Test Precision':<20} {'0.1158':>10}  {'0.1241':>10}  {test_prec:>10.4f}")
print(f"  {'Test Recall':<20} {'0.4360':>10}  {'0.4116':>10}  {test_rec:>10.4f}")
print(f"{'='*60}")
print()
print(classification_report(y_test, y_pred_test,
                             target_names=["正常", "黑名單"], digits=4))

# 儲存最佳模型
with open(f"{OUTPUT_DIR}/next_step_model.pkl", "wb") as f:
    pickle.dump({
        "xgb": final_xgb, "lgbm": final_lgbm,
        "scaler": final_scaler, "threshold": best_thr,
        "features": FEAT_COLS, "best_spw": best_spw,
        "oof_f1": best_oof_f1, "test_f1": test_f1,
    }, f)

print(f"模型已儲存: {OUTPUT_DIR}/next_step_model.pkl")
print("\n把結果告訴我，我們決定要往哪個方向繼續深挖")