import pandas as pd
import numpy as np
import os, pickle, warnings
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, classification_report, precision_recall_curve
from sklearn.preprocessing import StandardScaler
from imblearn.under_sampling import RandomUnderSampler
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

warnings.filterwarnings("ignore")

FEAT_DIR = "./data/features"
OUTPUT_DIR = "./data/model_output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 60)
print("BitoGuard 模型訓練（F1-score 衝刺版）")
print("=" * 60)

df = pd.read_parquet(f"{FEAT_DIR}/feature_matrix.parquet")
print(f"特徵矩陣: {df.shape[0]} 用戶 × {df.shape[1]} 欄位")
print(f"黑名單比例: {df['status'].mean()*100:.2f}%")

df_enhanced = df.copy()
df_enhanced['risk_index'] = (df['twd_withdraw_ratio'] * df['twd_is_quick_out']) + df['has_blacklist_contact_1hop']
df_enhanced['ip_intensity'] = df['twd_unique_ip'] + df['crypto_unique_ip']

FEAT_COLS = [c for c in df_enhanced.columns if c not in ["user_id", "status"]]
X = df_enhanced[FEAT_COLS].values.astype(float)
y = df_enhanced["status"].values.astype(int)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

scale_pos = int((y == 0).sum() / max((y == 1).sum(), 1))
balanced_weight = int(scale_pos ** 0.5)

print(f"原始正負樣本比: {scale_pos}:1")
print(f"平衡權重: {balanced_weight}")

def make_xgb():
    return XGBClassifier(n_estimators=400, max_depth=5, learning_rate=0.03,
                         subsample=0.8, colsample_bytree=0.7,
                         scale_pos_weight=balanced_weight, random_state=42,
                         eval_metric="aucpr", use_label_encoder=False, verbosity=0)

def make_lgbm():
    return LGBMClassifier(n_estimators=400, max_depth=5, learning_rate=0.03,
                          subsample=0.8, colsample_bytree=0.7,
                          scale_pos_weight=balanced_weight, random_state=42, verbose=-1)

sampler = RandomUnderSampler(sampling_strategy=0.1, random_state=42)

print("\n[1] 執行 5-Fold 交叉驗證...")
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_results = []
best_thresholds = []

for fold, (train_idx, val_idx) in enumerate(skf.split(X_scaled, y), 1):
    X_tr, X_val = X_scaled[train_idx], X_scaled[val_idx]
    y_tr, y_val = y[train_idx], y[val_idx]
    X_tr_res, y_tr_res = sampler.fit_resample(X_tr, y_tr)
    
    xgb = make_xgb()
    lgbm = make_lgbm()
    xgb.fit(X_tr_res, y_tr_res, eval_set=[(X_val, y_val)], verbose=False)
    lgbm.fit(X_tr_res, y_tr_res)
    
    prob_xgb = xgb.predict_proba(X_val)[:, 1]
    prob_lgbm = lgbm.predict_proba(X_val)[:, 1]
    y_prob = (prob_xgb * 0.7 + prob_lgbm * 0.3)
    
    prec, rec, thrs = precision_recall_curve(y_val, y_prob)
    f1s = 2 * prec * rec / (prec + rec + 1e-9)
    best_idx = np.argmax(f1s[:-1])
    best_thr = thrs[best_idx]
    best_thresholds.append(best_thr)
    
    y_pred = (y_prob >= best_thr).astype(int)
    p = precision_score(y_val, y_pred, zero_division=0)
    r = recall_score(y_val, y_pred, zero_division=0)
    f1 = f1_score(y_val, y_pred, zero_division=0)
    
    cv_results.append({"precision": p, "recall": r, "f1": f1})
    print(f"  Fold {fold}: F1={f1:.4f}  P={p:.4f}  R={r:.4f}  Thr={best_thr:.3f}")

final_threshold = float(np.mean(best_thresholds))
print(f"\n平均 F1: {np.mean([m['f1'] for m in cv_results]):.4f}")

print("\n[2] 訓練最終模型...")
X_res, y_res = sampler.fit_resample(X_scaled, y)
final_xgb = make_xgb()
final_lgbm = make_lgbm()
final_xgb.fit(X_res, y_res, verbose=False)
final_lgbm.fit(X_res, y_res)

y_prob_all = (final_xgb.predict_proba(X_scaled)[:, 1] * 0.7 + final_lgbm.predict_proba(X_scaled)[:, 1] * 0.3)
y_pred_all = (y_prob_all >= final_threshold).astype(int)

print("\n最終表現：")
print(classification_report(y, y_pred_all, target_names=["正常", "黑名單"]))

cm = confusion_matrix(y, y_pred_all)
tn, fp, fn, tp = cm.ravel()
print(f"TP={tp}  FP={fp}  FN={fn}  TN={tn}")

df_enhanced['risk_score'] = y_prob_all
df_enhanced['predicted'] = y_pred_all
result_df = df_enhanced[['user_id', 'risk_score', 'predicted', 'status']].copy()
result_df.to_csv(f"{OUTPUT_DIR}/predictions_final.csv", index=False)

with open(f"{OUTPUT_DIR}/final_model_final.pkl", "wb") as f:
    pickle.dump({"xgb": final_xgb, "lgbm": final_lgbm, "scaler": scaler, 
                 "threshold": final_threshold, "features": FEAT_COLS}, f)

final_f1 = f1_score(y, y_pred_all)
print(f"\n訓練完成！F1-score: {final_f1:.4f}")
print(f"檔案已儲存至 {OUTPUT_DIR}")
