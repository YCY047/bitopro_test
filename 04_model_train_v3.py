import pandas as pd
import numpy as np
import os, pickle, warnings
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (precision_score, recall_score, f1_score, 
                              roc_auc_score, precision_recall_curve, classification_report)
from sklearn.preprocessing import StandardScaler
from imblearn.under_sampling import RandomUnderSampler
from imblearn.ensemble import BalancedRandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

warnings.filterwarnings("ignore")

# 1. 環境與資料設定
FEAT_DIR = "./data/features"
OUTPUT_DIR = "./data/model_output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 60)
print("BitoGuard 訓練 V3：F1-score 極大化優化版")
print("=" * 60)

# 載入原始特徵矩陣
df = pd.read_parquet(f"{FEAT_DIR}/feature_matrix.parquet")
print(f"成功載入資料: {len(df)} 筆用戶紀錄")

# --- 進階特徵工程：針對 0.167 瓶頸進行交互計算 ---
print("[0] 正在生成深度風險交互特徵...")
df_v3 = df.copy()

# A. 快速進出強度：入金後立刻提領且金額比例高
df_v3['quick_exit_intensity'] = df['twd_is_quick_out'] * df['twd_withdraw_ratio'] * df['twd_tx_count']

# B. 深夜行為指標：法幣與虛幣皆在深夜操作
df_v3['night_owl_score'] = df['twd_night_ratio'] * df['crypto_night_ratio']

# C. 帳戶關聯度權重：將 1 層與 2 層關聯加權
df_v3['blacklist_network_risk'] = (df['has_blacklist_contact_1hop'] * 3) + (df['has_blacklist_contact_2hop'] * 1)

# D. 設備分散度：IP 數與交易次數的比例
df_v3['ip_per_tx'] = df['twd_unique_ip'] / (df['twd_tx_count'] + 1)

# E. 綜合風險指數
df_v3['composite_risk'] = df_v3['quick_exit_intensity'] + df_v3['blacklist_network_risk']

FEAT_COLS = [c for c in df_v3.columns if c not in ["user_id", "status"]]
X = df_v3[FEAT_COLS].values.astype(float)
y = df_v3["status"].values.astype(int)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 2. 模型權重設定
scale_pos = int((y == 0).sum() / max((y == 1).sum(), 1))
# 為了拉高 F1，權重不宜過高。使用 0.4 次方比開根號更保守，有助於 Precision
balanced_weight = int(scale_pos ** 0.4) 

# 3. 定義模型組合
def make_xgb():
    return XGBClassifier(n_estimators=400, max_depth=5, learning_rate=0.03,
                        scale_pos_weight=balanced_weight, eval_metric="aucpr",
                        subsample=0.8, colsample_bytree=0.7, random_state=42)

def make_lgbm():
    return LGBMClassifier(n_estimators=400, max_depth=5, learning_rate=0.03,
                         scale_pos_weight=balanced_weight, random_state=42, verbose=-1)

def make_rf():
    return BalancedRandomForestClassifier(n_estimators=200, max_depth=10, 
                                          sampling_strategy='auto', random_state=42)

# 4. 交叉驗證與閾值優化
print("\n[1] 執行 5-Fold 交叉驗證 (三模型融合投票)...")
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
sampler = RandomUnderSampler(sampling_strategy=0.15, random_state=42) # 提高正樣本比例至 1:6
best_thrs = []

for fold, (train_idx, val_idx) in enumerate(skf.split(X_scaled, y), 1):
    X_tr, X_val = X_scaled[train_idx], X_scaled[val_idx]
    y_tr, y_val = y[train_idx], y[val_idx]

    X_tr_res, y_tr_res = sampler.fit_resample(X_tr, y_tr)

    m1, m2, m3 = make_xgb(), make_lgbm(), make_rf()
    m1.fit(X_tr_res, y_tr_res)
    m2.fit(X_tr_res, y_tr_res)
    m3.fit(X_tr_res, y_tr_res)

    # 權重投票：XGB(50%) + LGBM(30%) + RF(20%)
    p1 = m1.predict_proba(X_val)[:, 1]
    p2 = m2.predict_proba(X_val)[:, 1]
    p3 = m3.predict_proba(X_val)[:, 1]
    y_prob = (p1 * 0.5 + p2 * 0.3 + p3 * 0.2)

    prec, rec, thrs = precision_recall_curve(y_val, y_prob)
    f1s = 2 * prec * rec / (prec + rec + 1e-9)
    best_thrs.append(thrs[np.argmax(f1s[:-1])])
    print(f"  Fold {fold} | 最高 F1: {np.max(f1s):.4f}")

final_thr = np.mean(best_thrs)

# 5. 訓練最終模型
print("\n[2] 訓練最終全量模型...")
X_res, y_res = sampler.fit_resample(X_scaled, y)
final_m1, final_m2, final_m3 = make_xgb(), make_lgbm(), make_rf()
final_m1.fit(X_res, y_res)
final_m2.fit(X_res, y_res)
final_m3.fit(X_res, y_res)

y_prob_all = (final_m1.predict_proba(X_scaled)[:, 1] * 0.5 + 
              final_m2.predict_proba(X_scaled)[:, 1] * 0.3 + 
              final_m3.predict_proba(X_scaled)[:, 1] * 0.2)
y_pred_all = (y_prob_all >= final_thr).astype(int)

# 儲存結果
df_v3['risk_score'] = y_prob_all
df_v3['predicted'] = y_pred_all
output_path = f"{OUTPUT_DIR}/predictions_v3.csv"
df_v3[['user_id', 'risk_score', 'predicted', 'status']].to_csv(output_path, index=False)

with open(f"{OUTPUT_DIR}/model_v3.pkl", "wb") as f:
    pickle.dump({"m1": final_m1, "m2": final_m2, "m3": final_m3, "scaler": scaler, "thr": final_thr}, f)

print("\n" + "=" * 60)
print(f"訓練完成！F1-score 最佳化閾值: {final_thr:.4f}")
print(f"報告已存至: {os.path.abspath(output_path)}")
print(classification_report(y, y_pred_all, target_names=["正常", "黑名單"]))