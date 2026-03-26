import pandas as pd
import requests
import os
from datetime import datetime

API_BASE = "https://aws-event-api.bitopro.com"
RAW_DIR = "./data/raw"
os.makedirs(RAW_DIR, exist_ok=True)

def fetch_table(endpoint, table_name, batch_size=1000):
    print(f"\n[抓取] {table_name} ...")
    all_data = []
    offset = 0
    
    while True:
        url = f"{API_BASE}{endpoint}"
        params = {"limit": batch_size, "offset": offset}
        
        try:
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()
            
            if not data or len(data) == 0:
                break
            
            all_data.extend(data)
            print(f"  已抓取 {len(all_data)} 筆...", end="\r")
            
            if len(data) < batch_size:
                break
            
            offset += batch_size
            
        except requests.exceptions.RequestException as e:
            print(f"\n    抓取失敗: {e}")
            if len(all_data) > 0:
                print(f"  已抓取 {len(all_data)} 筆，繼續處理...")
                break
            else:
                print(f"  無法抓取任何資料，跳過此表")
                return pd.DataFrame()
    
    if len(all_data) == 0:
        print(f"    {table_name} 無資料")
        return pd.DataFrame()
    
    df = pd.DataFrame(all_data)
    print(f"\n   {table_name}: {len(df)} 筆")
    return df

print("=" * 60)
print("BitoGuard 資料抓取")
print(f"API: {API_BASE}")
print(f"時間: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 60)

user_info = fetch_table("/user_info", "user_info")
train_label = fetch_table("/train_label", "train_label")

if not user_info.empty and not train_label.empty:
    print("\n[合併] user_info + train_label ...")
    user_info = user_info.merge(train_label, on="user_id", how="left")
    user_info["status"] = user_info["status"].fillna(0).astype(int)
    print(f"   合併完成，黑名單用戶: {(user_info['status']==1).sum()} 人")
elif not train_label.empty:
    print("\n    user_info 為空，使用 train_label 作為 user_info")
    user_info = train_label.copy()

if not user_info.empty:
    user_info.to_parquet(f"{RAW_DIR}/user_info.parquet", index=False)
    print(f"   儲存: {RAW_DIR}/user_info.parquet")

twd_transfer = fetch_table("/twd_transfer", "twd_transfer")
if not twd_transfer.empty:
    if "source_ip_hash" in twd_transfer.columns:
        twd_transfer = twd_transfer.rename(columns={"source_ip_hash": "source_ip"})
    twd_transfer.to_parquet(f"{RAW_DIR}/twd_transfer.parquet", index=False)
    print(f"   儲存: {RAW_DIR}/twd_transfer.parquet")

crypto_transfer = fetch_table("/crypto_transfer", "crypto_transfer")
if not crypto_transfer.empty:
    rename_map = {"source_ip_hash": "source_ip", "from_wallet_hash": "from_wallet", "to_wallet_hash": "to_wallet"}
    crypto_transfer = crypto_transfer.rename(columns=rename_map)
    crypto_transfer.to_parquet(f"{RAW_DIR}/crypto_transfer.parquet", index=False)
    print(f"   儲存: {RAW_DIR}/crypto_transfer.parquet")

usdt_twd_trading = fetch_table("/usdt_twd_trading", "usdt_twd_trading")
if not usdt_twd_trading.empty:
    if "source_ip_hash" in usdt_twd_trading.columns:
        usdt_twd_trading = usdt_twd_trading.rename(columns={"source_ip_hash": "source_ip"})
    usdt_twd_trading.to_parquet(f"{RAW_DIR}/usdt_twd_trading.parquet", index=False)
    print(f"   儲存: {RAW_DIR}/usdt_twd_trading.parquet")

usdt_swap = fetch_table("/usdt_swap", "usdt_swap")
if not usdt_swap.empty:
    usdt_swap.to_parquet(f"{RAW_DIR}/usdt_swap.parquet", index=False)
    print(f"   儲存: {RAW_DIR}/usdt_swap.parquet")

print("\n" + "=" * 60)
print("資料抓取完成！")
print(f"\n資料統計:")
if not user_info.empty:
    print(f"  user_info       : {len(user_info):>6} rows")
    if "status" in user_info.columns:
        bl_count = (user_info["status"] == 1).sum()
        bl_pct = bl_count / len(user_info) * 100 if len(user_info) > 0 else 0
        print(f"     黑名單     : {bl_count:>6} 人 ({bl_pct:.1f}%)")
if not twd_transfer.empty:
    print(f"  twd_transfer    : {len(twd_transfer):>6} rows")
if not crypto_transfer.empty:
    print(f"  crypto_transfer : {len(crypto_transfer):>6} rows")
if not usdt_twd_trading.empty:
    print(f"  usdt_twd_trading: {len(usdt_twd_trading):>6} rows")
if not usdt_swap.empty:
    print(f"  usdt_swap       : {len(usdt_swap):>6} rows")

print(f"\n所有檔案儲存於: {RAW_DIR}/")
print("\n接下來執行:")
print("  python 02_eda.py")
print("  python 03_feature_engineering.py")
print("  python 04_model_train.py")
print("=" * 60)
