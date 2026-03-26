"""
BitoGuard - Stacking 集成模型
使用多層模型堆疊來提升 F1-score
"""

import pandas as pd
import numpy as np
import os, json, pickle, warnings
import matplotlib
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")

# Windows 中文字體設定
matplotlib.rcParams["font.sans-serif"] = ["Microsoft JhengHei", "Microsoft YaHei", "SimHei"]
matplotlib.rcParams["axes.unicode_minus"] = False

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (precision_score, recall_score, f1_score,
                              confusion_matrix, classification_report,
                              roc_auc_score, precision_recall_curve)
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTETomek

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
import shap

FEAT_DIR   = "./data/features"
OUTPUT_DIR = "./data/model_output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 80)
print("BitoGuard - Stacking 集成模型")
print("=" * 80)

# ── 載入資料 ─────────────────────────────────────────────
df = pd.read_parquet(f"{FEAT_DIR}/feature_matrix.parquet")
print(f"\n資料: {df.shape[0]} 用戶 × {df.shape[1]} 欄位")
print(f"黑名單比例: {df['status'].mean()*100:.2f}%\n")

# ── 特徵工程增強 ──────────────────────────────────────────
print("[1] 特徵工程增強...")

df_enhanced = df.copy()

# 基礎交互特徵
df_enhanced['withdraw_x_night'] = df['twd_withdraw_ratio'] * df['twd_night_ratio']
df_enhanced['ip_x_withdraw'] = df['twd_unique_ip'] * df['twd_withdraw_ratio']
df_enhanced['quick_out_x_amount'] = df['twd_is_quick_out'] * df['twd_max_amount']
df_enhanced['blacklist_contact_any'] = (df['has_blacklist_contact_1hop'] + df['has_blacklist_contact_2hop']).clip(0, 1)

# 綜合特徵
df_enhanced['total_night_ratio'] = (df['twd_night_ratio'] + df['crypto_night_ratio'] + df['trade_night_ratio']) / 3
df_enhanced['total_unique_ip'] = df['twd_unique_ip'] + df['crypto_unique_ip'] + df['trade_unique_ip']
df_enhanced['activity_score'] = df['twd_tx_count'] + df['crypto_tx_count'] + df['trade_tx_count'] + df['swap_tx_count']

# 風險特徵
df_enhanced['kyc_risk'] = (df['kyc_l2_delay_days'] < 30).astype(int) + df['is_high_risk_career']
df_enhanced['behavior_risk'] = (df['twd_withdraw_ratio'] > 0.7).astype(int) + (df['twd_night_ratio'] > 0.3).astype(int)

print(f"   新增 9 個交互特徵")

FEAT_COLS = [c for c in df_enhanced.columns if c not in ["user_id", "status"]]
X = df_enhanced[FEAT_COLS].values.astype(float)
y = df_enhanced["status"].values.astype(int)

scale_pos = int((y == 0).sum() / max((y == 1).sum(), 1))

# ── 定義基礎模型（Level 0）──────────────────────────────
print("\n[2] 定義 Stacking 模型...")
print("   Level 0 (基礎模型): 5 個強分類器")
print("   Level 1 (元模型): Logistic Regression")

# Level 0: 基礎模型
base_models = [
    ('xgb', XGBClassifier(
        n_estimators=500, max_depth=8, learning_rate=0.03,
        subsample=0.8, colsample_bytree=0.8,
        scale_pos_weight=scale_pos * 2,
        min_child_weight=2, gamma=0.2,
        reg_alpha=0.1, reg_lambda=1.0,
        random_state=42, eval_metric="logloss",
        use_label_encoder=False, verbosity=0
    )),
    ('lgbm', LGBMClassifier(
        n_estimators=500, max_depth=8, learning_rate=0.03,
        subsample=0.8, colsample_bytree=0.8,
        scale_pos_weight=scale_pos * 2,
        min_child_samples=15,
        reg_alpha=0.1, reg_lambda=1.0,
        random_state=42, verbose=-1
    )),
    ('catboost', CatBoostClassifier(
        iterations=400, depth=7, learning_rate=0.04,
        scale_pos_weight=scale_pos * 2,
        l2_leaf_reg=3,
        random_state=42, verbose=0
    )),
    ('rf', RandomForestClassifier(
        n_estimators=300, max_depth=10,
        class_weight='balanced',
        min_samples_split=10,
        random_state=42, n_jobs=-1
    )),
    ('et', ExtraTreesClassifier(
        n_estimators=300, max_depth=10,
        class_weight='balanced',
        min_samples_split=10,
        random_state=42, n_jobs=-1
    )),
]

# Level 1: 元模型（使用 Logistic Regression）
meta_model = LogisticRegression(
    class_weight='balanced',
    max_iter=1000,
    random_state=42
)

# 創建 Stacking Classifier
stacking_model = StackingClassifier(
    estimators=base_models,
    final_estimator=meta_model,
    cv=3,  # 內部交叉驗證
    stack_method='predict_proba',  # 使用機率作為元特徵
    n_jobs=-1
)

# ── 交叉驗證 ──────────────────────────────────────────────
print("\n[3] Stratified 5-Fold 交叉驗證...")
print("   使用 SMOTETomek 採樣")

smote_tomek = SMOTETomek(random_state=42)
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

cv_results = []
best_thresholds = []

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
    print(f"\n   Fold {fold}/5:")
    X_tr, X_val = X[train_idx], X[val_idx]
    y_tr, y_val = y[train_idx], y[val_idx]
    
    # SMOTETomek 採樣
    print(f"      採樣前: {len(y_tr)} 樣本")
    X_tr_res, y_tr_res = smote_tomek.fit_resample(X_tr, y_tr)
    print(f"      採樣後: {len(y_tr_res)} 樣本")
    
    # 訓練 Stacking 模型
    print(f"      訓練 Stacking 模型...")
    stacking_model.fit(X_tr_res, y_tr_res)
    
    # 預測機率
    y_prob = stacking_model.predict_proba(X_val)[:, 1]
    
    # 閾值優化
    precisions, recalls, thresholds = precision_recall_curve(y_val, y_prob)
    f1s = 2 * precisions * recalls / (precisions + recalls + 1e-9)
    best_idx = np.argmax(f1s[:-1])
    best_thr = thresholds[best_idx]
    best_thresholds.append(best_thr)
    
    # 預測
    y_pred = (y_prob >= best_thr).astype(int)
    
    # 計算指標
    p = precision_score(y_val, y_pred, zero_division=0)
    r = recall_score(y_val, y_pred, zero_division=0)
    f1 = f1_score(y_val, y_pred, zero_division=0)
    auc = roc_auc_score(y_val, y_prob)
    
    cv_results.append({
        "fold": fold,
        "threshold": best_thr,
        "precision": p,
        "recall": r,
        "f1": f1,
        "auc": auc
    })
    
    print(f"      thr={best_thr:.3f}  P={p:.4f}  R={r:.4f}  F1={f1:.4f}  AUC={auc:.4f}")

cv_df = pd.DataFrame(cv_results)
final_threshold = float(np.mean(best_thresholds))

print(f"\n   平均 Precision : {cv_df['precision'].mean():.4f} ± {cv_df['precision'].std():.4f}")
print(f"   平均 Recall    : {cv_df['recall'].mean():.4f} ± {cv_df['recall'].std():.4f}")
print(f"   平均 F1-score  : {cv_df['f1'].mean():.4f} ± {cv_df['f1'].std():.4f}")
print(f"   平均 AUC       : {cv_df['auc'].mean():.4f} ± {cv_df['auc'].std():.4f}")
print(f"   最終使用閾值   : {final_threshold:.4f}")

# ── 訓練最終模型 ──────────────────────────────────────────
print("\n[4] 訓練最終 Stacking 模型...")

X_res, y_res = smote_tomek.fit_resample(X, y)
print(f"   採樣後: {len(y_res)} 樣本")

print("   訓練 Stacking 模型（5 個基礎模型 + 元模型）...")
final_stacking = StackingClassifier(
    estimators=base_models,
    final_estimator=meta_model,
    cv=3,
    stack_method='predict_proba',
    n_jobs=-1
)
final_stacking.fit(X_res, y_res)

# 預測
y_prob_all = final_stacking.predict_proba(X)[:, 1]
y_pred_final = (y_prob_all >= final_threshold).astype(int)

df_enhanced["risk_score"] = y_prob_all
df_enhanced["risk_label"] = y_pred_final
df_enhanced["is_blacklist"] = y

print(f"\n   最終模型表現（閾值 = {final_threshold:.4f}）:")
print(classification_report(y, y_pred_final,
      target_names=["正常用戶", "黑名單"], digits=4))

cm = confusion_matrix(y, y_pred_final)
tn, fp, fn, tp = cm.ravel()
fpr = fp / max(fp + tn, 1)
print(f"   混淆矩陣: TP={tp}  FP={fp}  FN={fn}  TN={tn}")
print(f"   False Positive Rate: {fpr:.4f}")

# ── PR 曲線 ───────────────────────────────────────────────
print("\n[5] 生成 PR 曲線...")

precisions_all, recalls_all, thr_all = precision_recall_curve(y, y_prob_all)
f1s_all = 2 * precisions_all * recalls_all / (precisions_all + recalls_all + 1e-9)
best_plot_idx = np.argmin(np.abs(thr_all - final_threshold))

fig, ax = plt.subplots(figsize=(7, 5))
ax.plot(recalls_all[:-1], precisions_all[:-1], "b-", linewidth=2, label="Stacking PR curve")
ax.scatter(recalls_all[best_plot_idx], precisions_all[best_plot_idx],
           color="red", zorder=5, s=80, label=f"閾值={final_threshold:.3f}")
ax.set_xlabel("Recall（召回率）")
ax.set_ylabel("Precision（精確率）")
ax.set_title("Precision-Recall Curve（Stacking Model）")
ax.legend()
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/pr_curve_stacking.png", dpi=150)
plt.close()
print(f"   → {OUTPUT_DIR}/pr_curve_stacking.png")

# ── SHAP 可解釋性（使用 XGBoost 基礎模型）────────────────
print("\n[6] 計算 SHAP 值（使用 XGBoost 基礎模型）...")

# 獲取 XGBoost 基礎模型
xgb_base = final_stacking.named_estimators_['xgb']

explainer = shap.TreeExplainer(xgb_base)
shap_values = explainer.shap_values(X)
shap_df = pd.DataFrame(shap_values, columns=FEAT_COLS)

# 中文特徵名稱
FEAT_ZH = {
    "sex": "性別", "career": "職業類別", "income_source": "收入來源",
    "user_source": "註冊管道", "kyc_l2_delay_days": "KYC L2 完成速度",
    "kyc_l1_to_l2_days": "KYC L1→L2 間隔", "is_high_risk_career": "高風險職業",
    "twd_withdraw_ratio": "法幣出金比率", "twd_night_ratio": "法幣深夜交易比例",
    "twd_min_stay_hours": "法幣最短滯留時間", "twd_unique_ip": "法幣操作不同 IP 數",
    "has_blacklist_contact_1hop": "與黑名單直接內轉", "has_blacklist_contact_2hop": "與黑名單 2 層關聯",
    "withdraw_x_night": "出金×深夜交互", "ip_x_withdraw": "IP×出金交互",
    "blacklist_contact_any": "黑名單關聯（任意）", "total_night_ratio": "總深夜交易比例",
    "total_unique_ip": "總不同 IP 數", "kyc_risk": "KYC 風險分數",
    "behavior_risk": "行為風險分數",
}

feat_zh_list = [FEAT_ZH.get(c, c) for c in FEAT_COLS]

# SHAP 重要性圖
plt.figure(figsize=(10, 8))
shap.summary_plot(shap_values, X, feature_names=feat_zh_list,
                  plot_type="bar", show=False, max_display=20)
plt.title("特徵重要性排名（SHAP - Stacking Model）")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/shap_importance_stacking.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"   → {OUTPUT_DIR}/shap_importance_stacking.png")

# ── 儲存結果 ──────────────────────────────────────────────
print("\n[7] 儲存結果...")

# 預測結果
result_df = df_enhanced[["user_id", "risk_score", "risk_label", "status"]].copy()
result_df.columns = ["user_id", "risk_score", "predicted_blacklist", "actual_blacklist"]
result_df = result_df.sort_values("risk_score", ascending=False)
result_df.to_csv(f"{OUTPUT_DIR}/predictions_stacking.csv", index=False)
print(f"   → {OUTPUT_DIR}/predictions_stacking.csv")

# 模型
with open(f"{OUTPUT_DIR}/final_model_stacking.pkl", "wb") as f:
    pickle.dump({
        "stacking_model": final_stacking,
        "threshold": final_threshold,
        "features": FEAT_COLS,
        "feat_zh": FEAT_ZH,
    }, f)
print(f"   → {OUTPUT_DIR}/final_model_stacking.pkl")

# ── 最終摘要 ─────────────────────────────────────────────
final_p = precision_score(y, y_pred_final, zero_division=0)
final_r = recall_score(y, y_pred_final, zero_division=0)
final_f1 = f1_score(y, y_pred_final, zero_division=0)

print("\n" + "=" * 80)
print("訓練完成！")
print("=" * 80)

print(f"\nStacking 模型架構:")
print(f"  Level 0 (基礎模型):")
print(f"    • XGBoost (n_estimators=500, max_depth=8)")
print(f"    • LightGBM (n_estimators=500, max_depth=8)")
print(f"    • CatBoost (iterations=400, depth=7)")
print(f"    • RandomForest (n_estimators=300, max_depth=10)")
print(f"    • ExtraTrees (n_estimators=300, max_depth=10)")
print(f"  Level 1 (元模型):")
print(f"    • Logistic Regression (class_weight='balanced')")

print(f"\n交叉驗證結果:")
print(f"  CV 平均 F1    : {cv_df['f1'].mean():.4f} ± {cv_df['f1'].std():.4f}")
print(f"  CV 平均 AUC   : {cv_df['auc'].mean():.4f}")

print(f"\n全量資料表現:")
print(f"  Precision   : {final_p:.4f}")
print(f"  Recall      : {final_r:.4f}")
print(f"  F1-score    : {final_f1:.4f}")
print(f"  FP Rate     : {fpr:.4f}")

# 與原始模型比較
original_f1 = 0.1249
improvement = (final_f1 / original_f1 - 1) * 100

if improvement > 5:
    print(f"\n🎉 相比原始模型 (F1=0.1249) 提升了 {improvement:.1f}%！")
elif improvement > 0:
    print(f"\n✓ 相比原始模型 (F1=0.1249) 提升了 {improvement:.1f}%")
else:
    print(f"\n⚠️  與原始模型 (F1=0.1249) 相比: {improvement:+.1f}%")

print("\n" + "=" * 80)
