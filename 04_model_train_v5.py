"""
BitoGuard - Step 4: 模型訓練 V5
基於 V3 改進，針對真實資料重新設計特徵：
- 黑名單聯繫只有 5.4% 用戶有，移除作為主要特徵
- 強化 swap / crypto / twd 行為特徵
- 加入「有無交易」的二元特徵
- 修正 StandardScaler 資訊洩漏
"""

import pandas as pd
import numpy as np
import os, json, pickle, warnings
import matplotlib
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")
matplotlib.rcParams["font.sans-serif"] = ["Microsoft JhengHei", "Microsoft YaHei", "SimHei"]
matplotlib.rcParams["axes.unicode_minus"] = False

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (precision_score, recall_score, f1_score,
                              confusion_matrix, classification_report,
                              roc_auc_score, precision_recall_curve)
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
import shap

FEAT_DIR   = "./data/features"
OUTPUT_DIR = "./data/model_output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 60)
print("BitoGuard V5 — swap/crypto 行為強化版")
print("=" * 60)

df = pd.read_parquet(f"{FEAT_DIR}/feature_matrix.parquet")
n_black = (df['status'] == 1).sum()
n_total = len(df)
print(f"總用戶: {n_total:,}  黑名單: {n_black:,} ({n_black/n_total*100:.2f}%)\n")

# ══════════════════════════════════════════════════════
# 特徵工程（針對真實資料統計結果）
# ══════════════════════════════════════════════════════
print("[0] 特徵工程...")
d = df.copy()

# ── swap 特徵（黑名單 49.9% vs 正常 22.4%，最強信號）
d['swap_active']          = (d['swap_tx_count'] > 0).astype(int)
d['swap_log_total']       = np.log1p(d['swap_total_twd'])
d['swap_log_max']         = np.log1p(d['swap_max_twd'])
d['swap_log_count']       = np.log1p(d['swap_tx_count'])
d['swap_amount_per_tx']   = d['swap_total_twd'] / (d['swap_tx_count'] + 1)
d['swap_high_total']      = (d['swap_total_twd'] > 50000).astype(int)
d['swap_very_high_total'] = (d['swap_total_twd'] > 200000).astype(int)
d['swap_high_single']     = (d['swap_max_twd'] > 30000).astype(int)
d['swap_sell_heavy']      = (d['swap_sell_ratio'] > 0.5).astype(int)

# ── crypto 特徵（黑名單 91.2% vs 正常 79.4%）
d['crypto_active']        = (d['crypto_tx_count'] > 0).astype(int)
d['crypto_log_count']     = np.log1p(d['crypto_tx_count'])
d['crypto_log_withdraw']  = np.log1p(d['crypto_withdraw_count'])
d['crypto_heavy_user']    = (d['crypto_tx_count'] > 10).astype(int)
d['crypto_withdraw_heavy']= (d['crypto_withdraw_ratio'] > 0.7).astype(int)

# ── twd 特徵（黑名單 85.4% vs 正常 88.1%，差異小但組合有用）
d['twd_active']           = (d['twd_tx_count'] > 0).astype(int)
d['twd_log_amount']       = np.log1p(d.get('twd_total_amount', pd.Series(0, index=d.index)))
d['twd_log_max']          = np.log1p(d.get('twd_max_amount', pd.Series(0, index=d.index)))
d['twd_withdraw_heavy']   = (d['twd_withdraw_ratio'] > 0.8).astype(int)
d['twd_quick_out']        = d['twd_is_quick_out']

# ── 組合特徵（swap 活躍 + 大量出金）
d['swap_active_x_withdraw']    = d['swap_active'] * d['twd_withdraw_ratio']
d['swap_active_x_crypto']      = d['swap_active'] * d['crypto_active']
d['high_swap_x_withdraw']      = d['swap_high_total'] * d['twd_withdraw_heavy']
d['swap_log_x_crypto_log']     = d['swap_log_total'] * d['crypto_log_count']
d['crypto_withdraw_x_swap']    = d['crypto_withdraw_ratio'] * d['swap_log_total']

# ── KYC 特徵
d['kyc_fast']             = (d.get('kyc_l2_delay_days', pd.Series(999, index=d.index)) < 30).astype(int)
d['high_risk_career']     = d.get('is_high_risk_career', pd.Series(0, index=d.index))

# ── 黑名單聯繫（只有 5.4% 有，但有的話信號很強，保留但降低權重）
d['blacklist_contact']    = ((d['has_blacklist_contact_1hop'] + d['has_blacklist_contact_2hop']) > 0).astype(int)

# ── 排除在真實資料中無效的特徵
EXCLUDE = [
    'trade_night_ratio',    # 黑名單反而低 0.36x
    'trade_buy_ratio',      # 0.64x
    'trade_tx_count',       # 0.60x
    'trade_unique_ip',      # 0.85x
    'crypto_night_ratio',   # 0.63x
    'twd_night_ratio',      # 需確認，可能也無效
    'has_blacklist_contact_1hop',  # 已合併進 blacklist_contact
    'has_blacklist_contact_2hop',  # 已合併進 blacklist_contact
]

FEAT_COLS = [c for c in d.columns
             if c not in ['user_id', 'status'] + EXCLUDE]

print(f"  特徵總數: {len(FEAT_COLS)}")

# 印出 Top 特徵區分力
bl_d = d[d['status'] == 1]
nm_d = d[d['status'] == 0]
ratios = {}
for c in FEAT_COLS:
    b = bl_d[c].mean()
    n = nm_d[c].mean()
    ratios[c] = b / (n + 1e-9)

top_feats = sorted(ratios.items(), key=lambda x: x[1], reverse=True)[:20]
print("\n  Top 20 特徵（黑名單/正常 倍數）：")
for feat, ratio in top_feats:
    bar = "█" * min(int(ratio * 2), 25)
    print(f"    {feat:<35} {ratio:>6.2f}x  {bar}")

X = d[FEAT_COLS].fillna(0).values.astype(float)
y = d['status'].values.astype(int)
scale_pos = int((y == 0).sum() / max((y == 1).sum(), 1))
print(f"\n  scale_pos_weight: {scale_pos}")

# ══════════════════════════════════════════════════════
# 5-Fold CV — 同時測試不同 scale_pos_weight
# ══════════════════════════════════════════════════════
print("\n[1] 5-Fold CV（測試不同 scale_pos_weight）...")

smote = SMOTE(random_state=42, k_neighbors=5)
skf   = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

def run_cv(spw_multiplier=1.0):
    spw = int(scale_pos * spw_multiplier)
    fold_f1s, fold_thrs = [], []

    for fold, (tr_idx, val_idx) in enumerate(skf.split(X, y), 1):
        X_tr, X_val = X[tr_idx], X[val_idx]
        y_tr, y_val = y[tr_idx], y[val_idx]

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
            scale_pos_weight=spw, random_state=42,
            eval_metric="logloss", use_label_encoder=False, verbosity=0,
        )
        lgbm = LGBMClassifier(
            n_estimators=500, max_depth=6, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            scale_pos_weight=spw, random_state=42, verbose=-1,
        )
        xgb.fit(X_res, y_res, eval_set=[(X_val_s, y_val)], verbose=False)
        lgbm.fit(X_res, y_res)

        prob = (xgb.predict_proba(X_val_s)[:, 1] * 0.6 +
                lgbm.predict_proba(X_val_s)[:, 1] * 0.4)

        prec, rec, thrs = precision_recall_curve(y_val, prob)
        f1s = 2 * prec * rec / (prec + rec + 1e-9)
        best_idx = np.argmax(f1s[:-1])
        fold_f1s.append(f1s[best_idx])
        fold_thrs.append(float(thrs[best_idx]))

    return np.mean(fold_f1s), np.std(fold_f1s), np.mean(fold_thrs)

# 測試不同倍率
best_spw_mult = 1.0
best_cv_f1    = 0
print(f"  {'倍率':>6}  {'scale_pos':>10}  {'CV F1':>8}  {'std':>6}")
for mult in [0.5, 1.0, 1.5, 2.0, 3.0]:
    mean_f1, std_f1, mean_thr = run_cv(mult)
    mark = ""
    if mean_f1 > best_cv_f1:
        best_cv_f1    = mean_f1
        best_spw_mult = mult
        mark = " ← 最佳"
    print(f"  {mult:>6.1f}  {int(scale_pos*mult):>10}  {mean_f1:>8.4f}  {std_f1:>6.4f}{mark}")

print(f"\n  最佳倍率: {best_spw_mult}x  CV F1: {best_cv_f1:.4f}")

# ══════════════════════════════════════════════════════
# 最終模型（用最佳 scale_pos_weight）
# ══════════════════════════════════════════════════════
print("\n[2] 訓練最終模型...")
best_spw = int(scale_pos * best_spw_mult)

_, _, final_thr = run_cv(best_spw_mult)

final_scaler = StandardScaler()
X_s = final_scaler.fit_transform(X)
try:
    X_res, y_res = smote.fit_resample(X_s, y)
except Exception:
    X_res, y_res = X_s, y

final_xgb = XGBClassifier(
    n_estimators=500, max_depth=6, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8,
    scale_pos_weight=best_spw, random_state=42,
    eval_metric="logloss", use_label_encoder=False, verbosity=0,
)
final_lgbm = LGBMClassifier(
    n_estimators=500, max_depth=6, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8,
    scale_pos_weight=best_spw, random_state=42, verbose=-1,
)
final_xgb.fit(X_res, y_res, verbose=False)
final_lgbm.fit(X_res, y_res)

y_prob = (final_xgb.predict_proba(X_s)[:, 1] * 0.6 +
          final_lgbm.predict_proba(X_s)[:, 1] * 0.4)

# 閾值敏感度表
print("\n[3] 閾值敏感度分析:")
print(f"  {'閾值':>5}  {'Precision':>10}  {'Recall':>8}  {'F1':>8}  {'預測黑名單':>10}")
best_thr, best_f1_val = final_thr, 0
for thr in np.arange(0.05, 0.96, 0.05):
    yp = (y_prob >= thr).astype(int)
    p_ = precision_score(y, yp, zero_division=0)
    r_ = recall_score(y, yp, zero_division=0)
    f_ = f1_score(y, yp, zero_division=0)
    n_ = int(yp.sum())
    if f_ > best_f1_val:
        best_f1_val = f_
        best_thr    = thr
    mark = " ★" if f_ == best_f1_val else ""
    print(f"  {thr:.2f}  {p_:>10.4f}  {r_:>8.4f}  {f_:>8.4f}  {n_:>10}{mark}")

y_pred = (y_prob >= best_thr).astype(int)
final_p  = precision_score(y, y_pred, zero_division=0)
final_r  = recall_score(y, y_pred, zero_division=0)
final_f1 = f1_score(y, y_pred, zero_division=0)
cm = confusion_matrix(y, y_pred)
tn, fp, fn, tp = cm.ravel()

print(f"\n  最佳閾值: {best_thr:.2f}")
print(classification_report(y, y_pred, target_names=["正常", "黑名單"], digits=4))
print(f"  TP={tp}  FP={fp}  FN={fn}  TN={tn}")

# SHAP
print("\n[4] SHAP...")
try:
    explainer   = shap.TreeExplainer(final_xgb)
    shap_values = explainer.shap_values(X_s)
    shap_df     = pd.DataFrame(shap_values, columns=FEAT_COLS)
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X_s, feature_names=FEAT_COLS,
                      plot_type="bar", show=False, max_display=20)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/shap_importance.png", dpi=150, bbox_inches="tight")
    plt.close()
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X_s, feature_names=FEAT_COLS,
                      show=False, max_display=20)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/shap_beeswarm.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  完成")
except Exception as e:
    print(f"  失敗: {e}")
    shap_df = pd.DataFrame(np.zeros((len(X_s), len(FEAT_COLS))), columns=FEAT_COLS)

# 診斷書
print("\n[5] 風險診斷書...")
d['risk_score'] = y_prob
d['risk_label'] = y_pred
high_risk = d[d['risk_label'] == 1].copy()
reports = []

for i, (_, row) in enumerate(high_risk.iterrows()):
    uid   = int(row['user_id'])
    score = float(row['risk_score'])
    iloc_pos = high_risk.index.get_loc(_) if hasattr(high_risk.index, 'get_loc') else i

    try:
        user_shap = shap_df.loc[_] if _ in shap_df.index else shap_df.iloc[min(i, len(shap_df)-1)]
        top5 = user_shap.abs().sort_values(ascending=False).head(5)
    except Exception:
        top5 = pd.Series(dtype=float)

    sentences, reasons = [], []
    for feat, shap_val in top5.items():
        val = float(row.get(feat, 0))
        if feat == 'swap_active' and val > 0:
            s = f"有一鍵買賣交易（總額 {row.get('swap_total_twd',0):,.0f} TWD）"
        elif feat in ('swap_log_total', 'swap_total_twd', 'swap_high_total', 'swap_very_high_total'):
            s = f"一鍵買賣總金額高（{row.get('swap_total_twd',0):,.0f} TWD）"
        elif feat in ('swap_log_max', 'swap_max_twd', 'swap_high_single'):
            s = f"一鍵買賣單筆最大 {row.get('swap_max_twd',0):,.0f} TWD"
        elif feat == 'swap_log_count' and val > 0:
            s = f"一鍵買賣 {int(row.get('swap_tx_count',0))} 次"
        elif feat == 'blacklist_contact' and val > 0:
            s = "與已知黑名單帳戶有資金關聯"
        elif feat == 'twd_withdraw_ratio' and val > 0.7:
            s = f"法幣出金比率 {val*100:.0f}%"
        elif feat == 'crypto_withdraw_ratio' and val > 0.6:
            s = f"虛幣出金比率 {val*100:.0f}%"
        elif feat == 'crypto_log_count' and val > 0:
            s = f"虛幣交易 {int(row.get('crypto_tx_count',0))} 次（活躍度高）"
        elif feat == 'twd_is_quick_out' and val > 0:
            s = "法幣入金後快速出金（2小時內）"
        else:
            s = f"{feat} 異常（值={val:.3f}）"
        sentences.append(s)
        reasons.append({"feature": feat, "value": round(val, 4),
                        "shap": round(float(shap_val), 4), "description": s})

    level   = "極高風險" if score >= 0.8 else "高風險" if score >= 0.5 else "中風險"
    summary = (f"【{level}】用戶 {uid} 風險分數 {score:.1%}。"
               f"主要異常：{'；'.join(sentences[:3])}。")
    reports.append({"user_id": uid, "risk_score": round(score, 4),
                    "risk_level": level, "summary": summary, "top_reasons": reasons})

reports.sort(key=lambda x: x["risk_score"], reverse=True)
with open(f"{OUTPUT_DIR}/risk_diagnosis.json", "w", encoding="utf-8") as f:
    json.dump(reports, f, ensure_ascii=False, indent=2)
print(f"  診斷書：{len(reports)} 人")

# 儲存
result_df = d[["user_id","risk_score","risk_label","status"]].copy()
result_df.columns = ["user_id","risk_score","predicted_blacklist","actual_blacklist"]
result_df.sort_values("risk_score", ascending=False).to_csv(
    f"{OUTPUT_DIR}/predictions.csv", index=False)

with open(f"{OUTPUT_DIR}/final_model.pkl", "wb") as f:
    pickle.dump({"xgb": final_xgb, "lgbm": final_lgbm, "scaler": final_scaler,
                 "threshold": best_thr, "features": FEAT_COLS}, f)

print("\n" + "=" * 60)
print(f"訓練完成！")
print(f"  最佳閾值  : {best_thr:.2f}")
print(f"  Precision : {final_p:.4f}")
print(f"  Recall    : {final_r:.4f}")
print(f"  F1-score  : {final_f1:.4f}")
print(f"  TP={tp}  FP={fp}  FN={fn}  TN={tn}")
print("=" * 60)
print("\n接著: streamlit run 05_dashboard.py")
print("最後: python 06_submit.py")