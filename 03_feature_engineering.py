"""
BitoGuard - Step 3: 特徵工程（修正版）
"""

import pandas as pd
import numpy as np
import os
from functools import reduce

RAW_DIR  = "./data/raw"
FEAT_DIR = "./data/features"
os.makedirs(FEAT_DIR, exist_ok=True)

def load(name):
    path = f"{RAW_DIR}/{name}.parquet"
    if not os.path.exists(path):
        print(f"  [SKIP] {path} 不存在")
        return pd.DataFrame()
    df = pd.read_parquet(path)
    for c in df.columns:
        if "created_at" in c or "updated_at" in c:
            df[c] = pd.to_datetime(df[c], errors="coerce")
    return df

print("Loading ...")
ui    = load("user_info")
twd   = load("twd_transfer")
cryp  = load("crypto_transfer")
trade = load("usdt_twd_trading")
swap  = load("usdt_swap")

# 金額換算（×1e-8）
for df, cols in [
    (twd,   ["ori_samount"]),
    (cryp,  ["ori_samount", "twd_srate"]),
    (trade, ["trade_samount", "twd_srate"]),
    (swap,  ["twd_samount", "currency_samount"]),
]:
    for c in cols:
        if not df.empty and c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce") * 1e-8

print("Building features ...")
blacklist_ids = set(ui[ui["status"] == 1]["user_id"]) if not ui.empty else set()
print(f"  已知黑名單用戶: {len(blacklist_ids)} 人")

all_feats = []

# ── 1. 用戶基本資料 ──────────────────────────────────────
if not ui.empty:
    f = ui[["user_id","status","sex","career","income_source","user_source"]].copy()
    for c in ["confirmed_at","level1_finished_at","level2_finished_at"]:
        if c in ui.columns:
            ui[c] = pd.to_datetime(ui[c], errors="coerce")
    if "confirmed_at" in ui.columns and "level2_finished_at" in ui.columns:
        f["kyc_l2_delay_days"] = (ui["level2_finished_at"] - ui["confirmed_at"]).dt.days
    if "level1_finished_at" in ui.columns and "level2_finished_at" in ui.columns:
        f["kyc_l1_to_l2_days"] = (ui["level2_finished_at"] - ui["level1_finished_at"]).dt.days
    f["is_high_risk_career"] = ui["career"].isin({22,23,24,25}).astype(int)
    base = f.fillna(0)
    print(f"  user_info 特徵: {len(f.columns)-2} 個")

# ── 2. 法幣（TWD）特徵 ───────────────────────────────────
if not twd.empty:
    rows = []
    for uid, grp in twd.groupby("user_id"):
        r = {"user_id": uid}
        r["twd_tx_count"]       = len(grp)
        r["twd_deposit_count"]  = (grp["kind"] == 0).sum()
        r["twd_withdraw_count"] = (grp["kind"] == 1).sum()
        r["twd_withdraw_ratio"] = r["twd_withdraw_count"] / max(r["twd_tx_count"], 1)

        if "ori_samount" in grp.columns:
            r["twd_total_amount"] = grp["ori_samount"].sum()
            r["twd_max_amount"]   = grp["ori_samount"].max()
            dep_sum = grp[grp["kind"]==0]["ori_samount"].sum()
            wit_sum = grp[grp["kind"]==1]["ori_samount"].sum()
            r["twd_net_amount"]   = dep_sum - wit_sum

        if "created_at" in grp.columns and grp["created_at"].notna().any():
            r["twd_night_ratio"] = grp["created_at"].dt.hour.between(0,5).mean()
            dep = grp[grp["kind"]==0]["created_at"].dropna()
            wit = grp[grp["kind"]==1]["created_at"].dropna()
            if len(dep) > 0 and len(wit) > 0:
                stays = []
                for dt in dep:
                    diffs = [(w - dt).total_seconds()/3600 for w in wit if w > dt]
                    if diffs:
                        stays.append(min(diffs))
                if stays:
                    r["twd_min_stay_hours"]  = min(stays)
                    r["twd_mean_stay_hours"] = np.mean(stays)
                    r["twd_is_quick_out"]    = int(min(stays) < 2)

        if "source_ip" in grp.columns:
            r["twd_unique_ip"] = grp["source_ip"].dropna().nunique()

        rows.append(r)

    f_twd = pd.DataFrame(rows)
    all_feats.append(f_twd)
    print(f"  twd 特徵: {len(f_twd.columns)-1} 個，用戶數: {len(f_twd)}")

# ── 3. 虛擬貨幣特徵 ──────────────────────────────────────
if not cryp.empty:
    # 預先計算黑名單的 2hop 鄰居
    bl_neighbors = set(
        cryp[cryp["user_id"].isin(blacklist_ids)]["relation_user_id"].dropna().astype(int)
    ) if "relation_user_id" in cryp.columns else set()

    rows = []
    for uid, grp in cryp.groupby("user_id"):
        r = {"user_id": uid}
        r["crypto_tx_count"]       = len(grp)
        r["crypto_withdraw_count"] = (grp["kind"] == 1).sum()
        r["crypto_withdraw_ratio"] = r["crypto_withdraw_count"] / max(r["crypto_tx_count"], 1)

        if "sub_kind" in grp.columns:
            r["crypto_internal_ratio"] = (grp["sub_kind"] == 1).mean()

        if "currency" in grp.columns:
            r["crypto_unique_currency"] = grp["currency"].nunique()

        if "relation_user_id" in grp.columns:
            related = set(grp["relation_user_id"].dropna().astype(int))
            r["has_blacklist_contact_1hop"] = int(bool(related & blacklist_ids))
            r["has_blacklist_contact_2hop"] = int(bool(related & bl_neighbors))

        if "created_at" in grp.columns and grp["created_at"].notna().any():
            r["crypto_night_ratio"] = grp["created_at"].dt.hour.between(0,5).mean()

        if "source_ip" in grp.columns:
            r["crypto_unique_ip"] = grp["source_ip"].dropna().nunique()

        rows.append(r)

    f_cryp = pd.DataFrame(rows)
    all_feats.append(f_cryp)
    print(f"  crypto 特徵: {len(f_cryp.columns)-1} 個，用戶數: {len(f_cryp)}")

# ── 4. 掛單交易特徵 ──────────────────────────────────────
if not trade.empty:
    rows = []
    for uid, grp in trade.groupby("user_id"):
        r = {"user_id": uid}
        r["trade_tx_count"]     = len(grp)
        r["trade_buy_ratio"]    = grp["is_buy"].mean() if "is_buy" in grp.columns else 0
        r["trade_market_ratio"] = grp["is_market"].mean() if "is_market" in grp.columns else 0
        if "source_ip" in grp.columns:
            r["trade_unique_ip"] = grp["source_ip"].dropna().nunique()
        tc = "updated_at" if "updated_at" in grp.columns else None
        if tc and grp[tc].notna().any():
            r["trade_night_ratio"] = grp[tc].dt.hour.between(0,5).mean()
        rows.append(r)
    f_trade = pd.DataFrame(rows)
    all_feats.append(f_trade)
    print(f"  trade 特徵: {len(f_trade.columns)-1} 個，用戶數: {len(f_trade)}")

# ── 5. 一鍵買賣特徵 ──────────────────────────────────────
if not swap.empty:
    rows = []
    for uid, grp in swap.groupby("user_id"):
        r = {"user_id": uid}
        r["swap_tx_count"]   = len(grp)
        r["swap_sell_ratio"] = (grp["kind"] == 1).mean() if "kind" in grp.columns else 0
        if "twd_samount" in grp.columns:
            r["swap_max_twd"]   = grp["twd_samount"].max()
            r["swap_total_twd"] = grp["twd_samount"].sum()
        rows.append(r)
    f_swap = pd.DataFrame(rows)
    all_feats.append(f_swap)
    print(f"  swap 特徵: {len(f_swap.columns)-1} 個，用戶數: {len(f_swap)}")

# ── 6. 組合所有特徵 ──────────────────────────────────────
feature_matrix = base.copy()
for f in all_feats:
    if not f.empty and "user_id" in f.columns:
        feature_matrix = feature_matrix.merge(f, on="user_id", how="left")

feature_matrix = feature_matrix.fillna(0)

# ── 儲存 ────────────────────────────────────────────────
out_path = f"{FEAT_DIR}/feature_matrix.parquet"
feature_matrix.to_parquet(out_path, index=False)

feat_cols = [c for c in feature_matrix.columns if c not in ["user_id","status"]]
print(f"\n✅ 特徵矩陣儲存: {out_path}")
print(f"   shape: {feature_matrix.shape}")
print(f"   特徵數: {len(feat_cols)} 個")
print("\n特徵列表:")
for i, c in enumerate(feat_cols, 1):
    print(f"  {i:2d}. {c}")
print("\n接著執行 04_model_train.py")