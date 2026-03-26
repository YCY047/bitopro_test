"""
BitoGuard - 優化 V4：整合圖特徵 + 時間行為特徵
================================================
新增：
  - graph_bl_neighbor_ratio  鄰居中黑名單比例（連續值，比 0/1 更強）
  - graph_danger_score       加權危險分數
  - graph_2hop_bl_ratio      二度鄰居黑名單比例
  - twd_active_days          法幣操作橫跨幾天
  - twd_burst_ratio          最忙一天佔總交易量比
  - all_channels_burst       所有管道都是短期密集操作
  ...等圖 + 時間特徵
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
print("BitoGuard 優化 V4 — 圖特徵 + 時間特徵整合版")
print("Previous best: OOF=0.1919 / Test=0.2107 / SPW=3")
print("=" * 60)

df = pd.read_parquet(f"{FEAT_DIR}/feature_matrix.parquet")
d = df.copy()

# ══════════════════════════════════════════════════════════
# 沿用 V3 特徵工程
# ══════════════════════════════════════════════════════════
d['swap_active']              = (d['swap_tx_count'] > 0).astype(int)
d['swap_log_total']           = np.log1p(d['swap_total_twd'])
d['swap_log_max']             = np.log1p(d['swap_max_twd'])
d['swap_log_count']           = np.log1p(d['swap_tx_count'])
d['swap_amount_per_tx']       = d['swap_total_twd'] / (d['swap_tx_count'] + 1)
d['swap_high_total']          = (d['swap_total_twd'] > 50000).astype(int)
d['swap_very_high_total']     = (d['swap_total_twd'] > 200000).astype(int)
d['swap_ultra_high']          = (d['swap_total_twd'] > 300000).astype(int)
d['swap_mega']                = (d['swap_total_twd'] > 500000).astype(int)
d['swap_concentration']       = d['swap_max_twd'] / (d['swap_total_twd'] + 1)
d['swap_sell_heavy']          = (d['swap_sell_ratio'] > 0.5).astype(int)
d['swap_avg_amount']          = d['swap_total_twd'] / (d['swap_tx_count'] + 1)
d['swap_avg_20k']             = (d['swap_avg_amount'] > 20000).astype(int)
d['swap_avg_50k']             = (d['swap_avg_amount'] > 50000).astype(int)
d['swap_tier1']               = (d['swap_total_twd'].between(10000, 50000)).astype(int)
d['swap_tier2']               = (d['swap_total_twd'].between(50000, 200000)).astype(int)
d['swap_tier3']               = (d['swap_total_twd'] > 200000).astype(int)
d['swap_freq_low']            = (d['swap_tx_count'].between(1, 3)).astype(int)
d['swap_freq_mid']            = (d['swap_tx_count'].between(4, 10)).astype(int)
d['swap_freq_high']           = (d['swap_tx_count'] > 10).astype(int)
d['swap_large_single']        = (d['swap_avg_amount'] > 30000).astype(int)

d['crypto_active']            = (d['crypto_tx_count'] > 0).astype(int)
d['crypto_log_count']         = np.log1p(d['crypto_tx_count'])
d['crypto_log_withdraw']      = np.log1p(d['crypto_withdraw_count'])
d['crypto_heavy_user']        = (d['crypto_tx_count'] > 10).astype(int)
d['crypto_withdraw_heavy']    = (d['crypto_withdraw_ratio'] > 0.7).astype(int)
d['crypto_withdraw_tier']     = pd.cut(
    d['crypto_withdraw_ratio'], bins=[-0.01, 0.3, 0.6, 0.8, 1.01],
    labels=[0, 1, 2, 3]).astype(float)

d['twd_active']               = (d['twd_tx_count'] > 0).astype(int)
d['twd_withdraw_heavy']       = (d['twd_withdraw_ratio'] > 0.8).astype(int)
d['twd_quick_out']            = d['twd_is_quick_out']
d['twd_log_amount']           = np.log1p(d.get('twd_total_amount', pd.Series(0, index=d.index)))
d['withdraw_ratio_tier1']     = (d['twd_withdraw_ratio'].between(0.5, 0.7)).astype(int)
d['withdraw_ratio_tier2']     = (d['twd_withdraw_ratio'].between(0.7, 0.9)).astype(int)
d['withdraw_ratio_tier3']     = (d['twd_withdraw_ratio'] > 0.9).astype(int)

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
d['all_channel_active']       = (
    (d['swap_tx_count'] > 2).astype(int) *
    (d['crypto_tx_count'] > 3).astype(int) *
    (d['twd_tx_count'] > 2).astype(int)
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
d['swap_tier3_x_crypto']      = d['swap_tier3'] * d['crypto_log_count']
d['swap_tier3_x_withdraw']    = d['swap_tier3'] * d['twd_withdraw_ratio']
d['swap_large_x_multi']       = d['swap_large_single'] * d['multi_channel']
d['swap_to_crypto_cashout']   = d['swap_tier2'] * d['crypto_withdraw_heavy']
d['swap_large_x_crypto_out']  = d['swap_log_total'] * d['crypto_withdraw_ratio']
d['no_bl_large_swap']         = (1 - d['blacklist_contact']) * d['swap_tier3']
d['no_bl_large_swap_withdraw']= (1 - d['blacklist_contact']) * d['swap_tier3'] * d['twd_withdraw_ratio']
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
d['kyc_fast']                 = (d.get('kyc_l2_delay_days',
                                 pd.Series(999, index=d.index)) < 7).astype(int)

# ══════════════════════════════════════════════════════════
# 新增：圖網路特徵（從 feature_matrix 直接讀取）
# ══════════════════════════════════════════════════════════
GRAPH_COLS = [c for c in df.columns if c.startswith('graph_')]

if GRAPH_COLS:
    print(f"\n[圖特徵] 偵測到 {len(GRAPH_COLS)} 個圖特徵，直接使用")

    # 圖特徵和行為特徵的交叉組合
    if 'graph_danger_score' in d.columns:
        d['graph_log_danger']         = np.log1p(d['graph_danger_score'])
        d['graph_danger_x_swap']      = d['graph_log_danger'] * d['swap_log_total']
        d['graph_danger_x_withdraw']  = d['graph_log_danger'] * d['twd_withdraw_ratio']
        d['graph_danger_x_quick_out'] = d['graph_danger_score'] * d['twd_is_quick_out']
        d['graph_high_x_swap_large']  = (d['graph_danger_score'] > 5).astype(int) * d['swap_tier3']

    if 'graph_bl_neighbor_ratio' in d.columns:
        d['graph_ratio_x_swap']       = d['graph_bl_neighbor_ratio'] * d['swap_log_total']
        d['graph_ratio_x_withdraw']   = d['graph_bl_neighbor_ratio'] * d['twd_withdraw_ratio']
        d['graph_high_ratio']         = (d['graph_bl_neighbor_ratio'] > 0.3).astype(int)

    # 印出圖特徵區分力
    bl = d[d['status'] == 1]
    nm = d[d['status'] == 0]
    print(f"\n  {'圖特徵名稱':<38} {'黑名單':>8}  {'正常':>8}  {'倍數':>6}")
    print("  " + "-" * 65)
    graph_show = GRAPH_COLS + [c for c in d.columns if c.startswith('graph_') and c not in GRAPH_COLS]
    for c in graph_show:
        if c in d.columns:
            b = bl[c].mean()
            n = nm[c].mean()
            ratio = b / (n + 1e-9)
            bar = "█" * min(int(ratio * 2), 20)
            print(f"  {c:<38} {b:>8.4f}  {n:>8.4f}  {ratio:>6.2f}x  {bar}")
else:
    print("\n[圖特徵] 未偵測到圖特徵（feature_matrix 無 graph_* 欄位）")
    print("  → 請先執行新版 03_feature_engineering.py 再重跑本腳本")

# ══════════════════════════════════════════════════════════
# 新增：時間行為特徵（從 feature_matrix 直接讀取）
# ══════════════════════════════════════════════════════════
TIME_COLS = [c for c in df.columns if any(
    keyword in c for keyword in
    ['active_days', 'tx_density', 'burst', 'max_daily', 'min_active']
)]

if TIME_COLS:
    print(f"\n[時間特徵] 偵測到 {len(TIME_COLS)} 個時間特徵，直接使用")

    # 時間特徵和行為特徵的交叉組合
    if 'twd_active_days' in d.columns:
        # 短期密集 + 大額 swap = 高危
        d['short_burst_x_swap']       = d['twd_is_short_burst'] * d['swap_log_total'] if 'twd_is_short_burst' in d.columns else 0
        d['short_burst_x_withdraw']   = d['twd_is_short_burst'] * d['twd_withdraw_ratio'] if 'twd_is_short_burst' in d.columns else 0
        # 活躍天數的 log（讓分佈更平滑）
        d['log_twd_active_days']      = np.log1p(d['twd_active_days'])

    if 'all_channels_burst' in d.columns:
        d['all_burst_x_swap']         = d['all_channels_burst'] * d['swap_log_total']
        d['all_burst_x_suspicious']   = d['all_channels_burst'] * d['suspicious_count']

    # 印出時間特徵區分力
    bl = d[d['status'] == 1]
    nm = d[d['status'] == 0]
    print(f"\n  {'時間特徵名稱':<38} {'黑名單':>8}  {'正常':>8}  {'倍數':>6}")
    print("  " + "-" * 65)
    time_show = TIME_COLS + [c for c in ['short_burst_x_swap', 'short_burst_x_withdraw',
                                          'all_burst_x_swap', 'all_burst_x_suspicious']
                              if c in d.columns]
    for c in time_show:
        if c in d.columns:
            b = bl[c].mean()
            n = nm[c].mean()
            ratio = b / (n + 1e-9)
            bar = "█" * min(int(ratio * 2), 20)
            print(f"  {c:<38} {b:>8.4f}  {n:>8.4f}  {ratio:>6.2f}x  {bar}")
else:
    print("\n[時間特徵] 未偵測到時間特徵")
    print("  → 請先執行新版 03_feature_engineering.py 再重跑本腳本")

# ══════════════════════════════════════════════════════════
# 建立特徵矩陣
# ══════════════════════════════════════════════════════════
EXCLUDE = [
    'trade_night_ratio', 'trade_buy_ratio', 'trade_tx_count',
    'trade_unique_ip', 'crypto_night_ratio',
    'has_blacklist_contact_1hop', 'has_blacklist_contact_2hop',
    # 時間原始欄位（first/last tx timestamp 不需要直接餵給模型）
    'twd_first_tx', 'twd_last_tx', 'cryp_first_tx', 'cryp_last_tx',
    'swap_first_tx', 'swap_last_tx',
]
FEAT_COLS = [c for c in d.columns if c not in ['user_id', 'status'] + EXCLUDE]
print(f"\n總特徵數: {len(FEAT_COLS)}")

# 印出所有特徵的區分力（Top 30）
bl = d[d['status'] == 1]
nm = d[d['status'] == 0]
ratios = {}
for c in FEAT_COLS:
    b = bl[c].mean() if c in bl.columns else 0
    n = nm[c].mean() if c in nm.columns else 0
    ratios[c] = b / (n + 1e-9)
top30 = sorted(ratios.items(), key=lambda x: x[1], reverse=True)[:30]
print(f"\nTop 30 特徵區分力：")
print(f"  {'特徵名稱':<40} {'倍數':>6}")
print("  " + "-" * 50)
for feat, ratio in top30:
    bar = "█" * min(int(ratio * 1.5), 20)
    print(f"  {feat:<40} {ratio:>6.2f}x  {bar}")

X = d[FEAT_COLS].fillna(0).values.astype(float)
y = d['status'].values.astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# ══════════════════════════════════════════════════════════
# CV 函數
# ══════════════════════════════════════════════════════════
def run_cv(X_train, y_train, spw, n_splits=5):
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
            n_estimators=600, max_depth=6, learning_rate=0.04,
            subsample=0.8, colsample_bytree=0.8, min_child_weight=3,
            scale_pos_weight=spw,
            random_state=42, eval_metric="logloss", verbosity=0,
        )
        lgbm = LGBMClassifier(
            n_estimators=600, max_depth=6, learning_rate=0.04,
            subsample=0.8, colsample_bytree=0.8, min_child_samples=15,
            scale_pos_weight=spw,
            random_state=42, verbose=-1,
        )
        xgb.fit(X_res, y_res, eval_set=[(X_val_s, y_val)], verbose=False)
        lgbm.fit(X_res, y_res)

        prob = (xgb.predict_proba(X_val_s)[:, 1] * 0.6 +
                lgbm.predict_proba(X_val_s)[:, 1] * 0.4)
        oof_probs[val_idx] = prob

    prec, rec, thrs = precision_recall_curve(y_train, oof_probs)
    f1s      = 2 * prec * rec / (prec + rec + 1e-9)
    best_idx = np.argmax(f1s[:-1])
    return float(f1s[best_idx]), float(thrs[best_idx]), oof_probs

# ══════════════════════════════════════════════════════════
# 搜尋最佳 SPW
# ══════════════════════════════════════════════════════════
print("\n" + "="*60)
print("搜尋最佳 SPW（1~15 細搜）")
print("="*60)
print(f"  {'SPW':>5}  {'OOF F1':>8}  {'閾值':>8}")

best_spw, best_oof_f1, best_thr = 3, 0, 0.5
best_oof_probs = None

for spw in [1, 2, 3, 5, 7, 9, 12, 15]:
    oof_f1, thr, oof_probs = run_cv(X_train, y_train, spw)
    mark = ""
    if oof_f1 > best_oof_f1:
        best_oof_f1    = oof_f1
        best_spw       = spw
        best_thr       = thr
        best_oof_probs = oof_probs
        mark           = " ← 最佳"
    print(f"  {spw:>5}  {oof_f1:>8.4f}  {thr:>8.4f}{mark}")

print(f"\n最佳 SPW={best_spw}  OOF F1={best_oof_f1:.4f}")

# ══════════════════════════════════════════════════════════
# 訓練最終模型
# ══════════════════════════════════════════════════════════
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
    n_estimators=600, max_depth=6, learning_rate=0.04,
    subsample=0.8, colsample_bytree=0.8, min_child_weight=3,
    scale_pos_weight=best_spw,
    random_state=42, eval_metric="logloss", verbosity=0,
)
final_lgbm = LGBMClassifier(
    n_estimators=600, max_depth=6, learning_rate=0.04,
    subsample=0.8, colsample_bytree=0.8, min_child_samples=15,
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
print("歷史對比")
print(f"{'='*60}")
print(f"  {'版本':<22} {'OOF F1':>8}  {'Test F1':>8}  {'Prec':>8}  {'Rec':>8}")
print(f"  {'-'*58}")
print(f"  {'Baseline':<22} {'0.1795':>8}  {'0.1830':>8}  {'0.1158':>8}  {'0.4360':>8}")
print(f"  {'Boost Features':<22} {'0.1804':>8}  {'0.1907':>8}  {'0.1241':>8}  {'0.4116':>8}")
print(f"  {'Next Step (SPW=9)':<22} {'0.1862':>8}  {'0.2072':>8}  {'0.1324':>8}  {'0.4756':>8}")
print(f"  {'V3 (SPW=3)':<22} {'0.1919':>8}  {'0.2107':>8}  {'0.1452':>8}  {'0.3841':>8}")
print(f"  {'V4 圖+時間（本版）':<22} {best_oof_f1:>8.4f}  {test_f1:>8.4f}  {test_prec:>8.4f}  {test_rec:>8.4f}")
print(f"{'='*60}")
print()
print(classification_report(y_test, y_pred_test,
                             target_names=["正常", "黑名單"], digits=4))

with open(f"{OUTPUT_DIR}/optimize_v4_model.pkl", "wb") as f:
    pickle.dump({
        "xgb": final_xgb, "lgbm": final_lgbm,
        "scaler": final_scaler, "threshold": best_thr,
        "features": FEAT_COLS, "best_spw": best_spw,
        "oof_f1": best_oof_f1, "test_f1": test_f1,
    }, f)

print(f"模型已儲存: {OUTPUT_DIR}/optimize_v4_model.pkl")
print("\n把 OOF F1 和 Test F1 告訴我")