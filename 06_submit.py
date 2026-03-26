"""
BitoGuard - Step 6: 提交預測結果到 /predict_label
使用方式: python 06_submit.py
"""

import requests
import pandas as pd
import json
import os

BASE_URL   = "https://aws-event-api.bitopro.com"
OUTPUT_DIR = "./data/model_output"
RAW_DIR    = "./data/raw"

HEADERS = {
    "Content-Type": "application/json",
    "Prefer": "resolution=merge-duplicates",
}

print("=" * 60)
print("BitoGuard 提交預測結果")
print("=" * 60)

# ── 讀取預測結果 ──────────────────────────────────────
pred_path = f"{OUTPUT_DIR}/predictions.csv"
if not os.path.exists(pred_path):
    print(f"[ERROR] 找不到 {pred_path}，請先執行 04_model_train.py")
    exit(1)

pred_df = pd.read_csv(pred_path)
print(f"預測結果: {len(pred_df):,} 筆")

# ── 只提交 predict_label 名單內的用戶 ─────────────────
pl_path = f"{RAW_DIR}/predict_label.parquet"
if os.path.exists(pl_path):
    predict_users = pd.read_parquet(pl_path)
    submit_df = pred_df[pred_df["user_id"].isin(predict_users["user_id"])].copy()
    print(f"需提交名單: {len(predict_users):,}  有預測結果: {len(submit_df):,}")
else:
    submit_df = pred_df.copy()
    print("[WARN] 未找到 predict_label.parquet，提交全部")

# ── 統計 ──────────────────────────────────────────────
n_black  = (submit_df["predicted_blacklist"] == 1).sum()
n_normal = (submit_df["predicted_blacklist"] == 0).sum()
print(f"\n預測黑名單: {n_black:,}")
print(f"預測正常:   {n_normal:,}")

# ── 建立 payload（只送黑名單 user_id）────────────────
blacklist_ids = submit_df[submit_df["predicted_blacklist"] == 1]["user_id"].tolist()
payload = [{"user_id": int(uid)} for uid in blacklist_ids]

print(f"\n準備 POST {len(payload):,} 筆至 {BASE_URL}/predict_label")
print("前 5 筆：", payload[:5])

confirm = input("\n確認提交？(y/n): ").strip().lower()
if confirm != "y":
    print("取消")
    exit(0)

# ── 提交 ─────────────────────────────────────────────
try:
    resp = requests.post(
        f"{BASE_URL}/predict_label",
        headers=HEADERS,
        data=json.dumps(payload),
        timeout=60,
    )
    print(f"\nHTTP {resp.status_code}")
    if resp.status_code in (200, 201, 204):
        print("✅ 提交成功！")
    else:
        print(f"回應: {resp.text[:500]}")
except Exception as e:
    print(f"[ERROR] {e}")

# 儲存提交記錄
submit_df[["user_id","risk_score","predicted_blacklist"]].to_csv(
    f"{OUTPUT_DIR}/submit_record.csv", index=False)
print(f"\n提交記錄: {OUTPUT_DIR}/submit_record.csv")