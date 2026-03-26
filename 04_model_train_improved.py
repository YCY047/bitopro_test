import sys
import os

# --- 強制最優先輸出，確認程式有活著 ---
print("\n" + "!"*30)
print("系統訊息：腳本已啟動！")
print("!"*30 + "\n")

try:
    import pandas as pd
    import numpy as np
    import pickle, warnings
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import f1_score, precision_recall_curve, classification_report
    from sklearn.preprocessing import StandardScaler
    from imblearn.under_sampling import RandomUnderSampler
    from imblearn.ensemble import BalancedRandomForestClassifier
    from xgboost import XGBClassifier
    from lightgbm import LGBMClassifier

    warnings.filterwarnings("ignore")

    # 設定路徑
    FEAT_PATH = "./data/features/feature_matrix.parquet"
    OUTPUT_DIR = "./data/model_output"
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 檢查原始資料是否存在
    if not os.path.exists(FEAT_PATH):
        print(f"❌ 錯誤：找不到特徵檔案 {FEAT_PATH}，請先執行 03 號腳本！")
        sys.exit()

    print(f"[1/5] 讀取資料中...")
    df = pd.read_parquet(FEAT_PATH)
    
    # 深度特徵交互 (V3 核心)
    df['risk_intensity'] = df['twd_is_quick_out'] * df['twd_withdraw_ratio']
    df['net_risk'] = (df['has_blacklist_contact_1hop'] * 5) + df['has_blacklist_contact_2hop']
    
    FEAT_COLS = [c for c in df.columns if c not in ["user_id", "status"]]
    X = df[FEAT_COLS].values.astype(float)
    y = df["status"].values.astype(int)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 權重與採樣設定
    scale_pos = int((y == 0).sum() / max((y == 1).sum(), 1))
    balanced_weight = int(scale_pos ** 0.45) 
    sampler = RandomUnderSampler(sampling_strategy=0.2, random_state=42) # 1:5 比例

    print(f"[2/5] 執行 5-Fold 交叉驗證 (F1 衝刺中)...")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    best_thrs = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X_scaled, y), 1):
        X_tr, X_val = X_scaled[train_idx], X_scaled[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]
        X_tr_res, y_tr_res = sampler.fit_resample(X_tr, y_tr)

        # 三模型投票
        m1 = XGBClassifier(n_estimators=300, max_depth=5, scale_pos_weight=balanced_weight, eval_metric="aucpr", random_state=42)
        m2 = LGBMClassifier(n_estimators=300, max_depth=5, scale_pos_weight=balanced_weight, random_state=42, verbose=-1)
        m3 = BalancedRandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)

        m1.fit(X_tr_res, y_tr_res)
        m2.fit(X_tr_res, y_tr_res)
        m3.fit(X_tr_res, y_tr_res)

        y_prob = (m1.predict_proba(X_val)[:, 1] * 0.5 + m2.predict_proba(X_val)[:, 1] * 0.3 + m3.predict_proba(X_val)[:, 1] * 0.2)
        
        prec, rec, thrs = precision_recall_curve(y_val, y_prob)
        f1s = 2 * prec * rec / (prec + rec + 1e-9)
        best_thrs.append(thrs[np.argmax(f1s[:-1])])
        print(f"      Fold {fold} | 當前 F1: {np.max(f1s):.4f}")

    final_thr = np.mean(best_thrs)

    print(f"[3/5] 訓練全量模型...")
    X_res, y_res = sampler.fit_resample(X_scaled, y)
    fm1, fm2, fm3 = m1, m2, m3 # 重用最後一折的模型參數
    fm1.fit(X_res, y_res); fm2.fit(X_res, y_res); fm3.fit(X_res, y_res)

    print(f"[4/5] 儲存結果檔案...")
    final_prob = (fm1.predict_proba(X_scaled)[:, 1] * 0.5 + fm2.predict_proba(X_scaled)[:, 1] * 0.3 + fm3.predict_proba(X_scaled)[:, 1] * 0.2)
    df['risk_score'] = final_prob
    df['predicted'] = (final_prob >= final_thr).astype(int)
    
    out_file = f"{OUTPUT_DIR}/predictions_v3.csv"
    df[['user_id', 'risk_score', 'predicted', 'status']].to_csv(out_file, index=False)
    
    print(f"[5/5] 任務完成！")
    print(f"✅ 檔案路徑: {os.path.abspath(out_file)}")
    print(classification_report(y, df['predicted'], target_names=["正常", "黑名單"]))

except Exception as e:
    print(f"\n❌ 發生程式錯誤：{str(e)}")
    import traceback
    traceback.print_exc()

input("\n請按 Enter 鍵結束視窗...")