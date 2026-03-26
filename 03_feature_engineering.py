"""
BitoGuard - Step 3: 特徵工程 V2
新增：
  - 圖網路特徵（鄰居黑名單比例、二度危險分數）
  - 時間行為特徵（活躍天數、交易密度、突發性）
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

def print_discrimination(df_feat, label_df, feat_cols, top_n=15):
    """印出特徵區分力（黑名單均值 / 正常均值）"""
    merged = df_feat.merge(label_df[["user_id", "status"]], on="user_id", how="left")
    bl = merged[merged["status"] == 1]
    nm = merged[merged["status"] == 0]
    rows = []
    for c in feat_cols:
        if c not in merged.columns:
            continue
        b = bl[c].mean()
        n = nm[c].mean()
        ratio = b / (n + 1e-9)
        rows.append((c, b, n, ratio))
    rows.sort(key=lambda x: x[3], reverse=True)
    print(f"  {'特徵名稱':<38} {'黑名單':>8}  {'正常':>8}  {'倍數':>6}")
    print("  " + "-" * 65)
    for c, b, n, ratio in rows[:top_n]:
        bar = "█" * min(int(ratio * 2), 20)
        print(f"  {c:<38} {b:>8.4f}  {n:>8.4f}  {ratio:>6.2f}x  {bar}")

print("=" * 60)
print("BitoGuard 特徵工程 V2")
print("=" * 60)

print("\nLoading ...")
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

print("\nBuilding features ...")
blacklist_ids = set(ui[ui["status"] == 1]["user_id"]) if not ui.empty else set()
print(f"  已知黑名單用戶: {len(blacklist_ids)} 人")

all_feats = []

# ══════════════════════════════════════════════════════════════
# 1. 用戶基本資料（KYC）
# ══════════════════════════════════════════════════════════════
if not ui.empty:
    f = ui[["user_id", "status", "sex", "career", "income_source", "user_source"]].copy()
    for c in ["confirmed_at", "level1_finished_at", "level2_finished_at"]:
        if c in ui.columns:
            ui[c] = pd.to_datetime(ui[c], errors="coerce")
    if "confirmed_at" in ui.columns and "level2_finished_at" in ui.columns:
        f["kyc_l2_delay_days"] = (ui["level2_finished_at"] - ui["confirmed_at"]).dt.days
    if "level1_finished_at" in ui.columns and "level2_finished_at" in ui.columns:
        f["kyc_l1_to_l2_days"] = (ui["level2_finished_at"] - ui["level1_finished_at"]).dt.days
    # KYC 完成速度：<7天 = 急著開始操作（可疑）
    f["kyc_very_fast"]       = (f.get("kyc_l2_delay_days", pd.Series(999, index=f.index)) < 7).astype(int)
    f["kyc_fast"]            = (f.get("kyc_l2_delay_days", pd.Series(999, index=f.index)) < 30).astype(int)
    f["is_high_risk_career"] = ui["career"].isin({22, 23, 24, 25}).astype(int)
    base = f.fillna(0)
    print(f"  [1] user_info 特徵: {len(f.columns)-2} 個")

# ══════════════════════════════════════════════════════════════
# 2. 法幣（TWD）特徵
# ══════════════════════════════════════════════════════════════
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
            dep_sum = grp[grp["kind"] == 0]["ori_samount"].sum()
            wit_sum = grp[grp["kind"] == 1]["ori_samount"].sum()
            r["twd_net_amount"]   = dep_sum - wit_sum

        if "created_at" in grp.columns and grp["created_at"].notna().any():
            r["twd_night_ratio"] = grp["created_at"].dt.hour.between(0, 5).mean()
            dep = grp[grp["kind"] == 0]["created_at"].dropna()
            wit = grp[grp["kind"] == 1]["created_at"].dropna()
            if len(dep) > 0 and len(wit) > 0:
                stays = []
                for dt in dep:
                    diffs = [(w - dt).total_seconds() / 3600 for w in wit if w > dt]
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
    print(f"  [2] twd 特徵: {len(f_twd.columns)-1} 個，用戶數: {len(f_twd)}")

# ══════════════════════════════════════════════════════════════
# 3. 虛擬貨幣特徵
# ══════════════════════════════════════════════════════════════
if not cryp.empty:
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
            r["crypto_night_ratio"] = grp["created_at"].dt.hour.between(0, 5).mean()

        if "source_ip" in grp.columns:
            r["crypto_unique_ip"] = grp["source_ip"].dropna().nunique()

        rows.append(r)

    f_cryp = pd.DataFrame(rows)
    all_feats.append(f_cryp)
    print(f"  [3] crypto 特徵: {len(f_cryp.columns)-1} 個，用戶數: {len(f_cryp)}")

# ══════════════════════════════════════════════════════════════
# 4. 掛單交易特徵
# ══════════════════════════════════════════════════════════════
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
            r["trade_night_ratio"] = grp[tc].dt.hour.between(0, 5).mean()
        rows.append(r)
    f_trade = pd.DataFrame(rows)
    all_feats.append(f_trade)
    print(f"  [4] trade 特徵: {len(f_trade.columns)-1} 個，用戶數: {len(f_trade)}")

# ══════════════════════════════════════════════════════════════
# 5. 一鍵買賣（Swap）特徵
# ══════════════════════════════════════════════════════════════
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
    print(f"  [5] swap 特徵: {len(f_swap.columns)-1} 個，用戶數: {len(f_swap)}")

# ══════════════════════════════════════════════════════════════
# 6. 圖網路特徵（NEW）
# 核心：把 has_blacklist_contact 從 0/1 升級成連續危險分數
# ══════════════════════════════════════════════════════════════
print("\n  [6] 圖網路特徵（建立轉帳圖）...")

if not cryp.empty and "relation_user_id" in cryp.columns and "sub_kind" in cryp.columns:

    # 只取平台內部轉帳（sub_kind==1 才有 relation_user_id）
    edges = (
        cryp[cryp["sub_kind"] == 1][["user_id", "relation_user_id"]]
        .dropna()
        .copy()
    )
    edges["relation_user_id"] = edges["relation_user_id"].astype(int)

    if len(edges) > 0:
        # ── 6a. 一度鄰居統計 ─────────────────────────────────
        neighbor_count = (
            edges.groupby("user_id")["relation_user_id"]
            .nunique()
            .reset_index(name="graph_neighbor_count")
        )

        edges["is_bl_neighbor"] = edges["relation_user_id"].isin(blacklist_ids).astype(int)
        bl_neighbor_count = (
            edges.groupby("user_id")["is_bl_neighbor"]
            .sum()
            .reset_index(name="graph_bl_neighbor_count")
        )

        graph_feat = neighbor_count.merge(bl_neighbor_count, on="user_id", how="left")
        graph_feat["graph_bl_neighbor_count"] = graph_feat["graph_bl_neighbor_count"].fillna(0)

        # 黑名單鄰居「比例」——比 0/1 更有區分力
        graph_feat["graph_bl_neighbor_ratio"] = (
            graph_feat["graph_bl_neighbor_count"] /
            graph_feat["graph_neighbor_count"].clip(lower=1)
        )

        # ── 6b. 二度鄰居危險分數 ─────────────────────────────
        print("    計算二度鄰居（可能需要 1~2 分鐘）...")
        user_to_neighbors = (
            edges.groupby("user_id")["relation_user_id"]
            .apply(set)
            .to_dict()
        )

        two_hop_bl   = {}
        two_hop_total = {}
        for uid, neighbors in user_to_neighbors.items():
            two_hop = set()
            for nb in neighbors:
                two_hop |= user_to_neighbors.get(nb, set())
            two_hop.discard(uid)
            two_hop -= neighbors  # 純二度，不含一度
            two_hop_total[uid] = len(two_hop)
            two_hop_bl[uid]    = len(two_hop & blacklist_ids)

        graph_feat["graph_2hop_total"]    = graph_feat["user_id"].map(two_hop_total).fillna(0)
        graph_feat["graph_2hop_bl_count"] = graph_feat["user_id"].map(two_hop_bl).fillna(0)
        graph_feat["graph_2hop_bl_ratio"] = (
            graph_feat["graph_2hop_bl_count"] /
            graph_feat["graph_2hop_total"].clip(lower=1)
        )

        # ── 6c. 綜合危險分數（加權組合）─────────────────────
        graph_feat["graph_danger_score"] = (
            graph_feat["graph_bl_neighbor_count"] * 3.0 +
            graph_feat["graph_2hop_bl_count"]     * 1.0 +
            graph_feat["graph_bl_neighbor_ratio"] * 10.0 +
            graph_feat["graph_2hop_bl_ratio"]     * 5.0
        )

        # ── 6d. 是否在黑名單密集子圖中 ──────────────────────
        graph_feat["graph_in_bl_cluster"] = (
            graph_feat["graph_bl_neighbor_ratio"] > 0.5
        ).astype(int)

        # ── 6e. log 版本（讓分佈更平滑）─────────────────────
        graph_feat["graph_log_danger"]    = np.log1p(graph_feat["graph_danger_score"])
        graph_feat["graph_log_bl_nb"]     = np.log1p(graph_feat["graph_bl_neighbor_count"])

        all_feats.append(graph_feat)

        n_graph_feats = len([c for c in graph_feat.columns if c != "user_id"])
        print(f"    圖特徵: {n_graph_feats} 個，有內轉用戶數: {len(graph_feat)}")

        print_discrimination(
            graph_feat, ui,
            [c for c in graph_feat.columns if c != "user_id"]
        )
    else:
        print("    [SKIP] 無內部轉帳資料（sub_kind==1 為空）")
else:
    print("    [SKIP] cryp 無 relation_user_id 或 sub_kind 欄位")

# ══════════════════════════════════════════════════════════════
# 7. 時間行為特徵（NEW）
# 核心：洗錢者通常「短時間密集操作然後消失」
# ══════════════════════════════════════════════════════════════
print("\n  [7] 時間行為特徵...")

time_feat_parts = []

# ── 7a. 法幣時間特徵 ─────────────────────────────────────────
if not twd.empty and "created_at" in twd.columns:
    twd_t = twd.dropna(subset=["created_at"]).copy()

    twd_time = (
        twd_t.groupby("user_id")["created_at"]
        .agg(twd_first_tx="min", twd_last_tx="max", twd_time_count="count")
        .reset_index()
    )
    twd_time["twd_active_days"] = (
        (twd_time["twd_last_tx"] - twd_time["twd_first_tx"])
        .dt.total_seconds() / 86400
    ).clip(lower=0)

    # 每天平均交易次數（密度越高越可疑）
    twd_time["twd_tx_density"] = (
        twd_time["twd_time_count"] / (twd_time["twd_active_days"] + 1)
    )

    # 短期突發（7天內完成所有操作）
    twd_time["twd_is_short_burst"] = (twd_time["twd_active_days"] <= 7).astype(int)
    twd_time["twd_is_very_short"]  = (twd_time["twd_active_days"] <= 1).astype(int)

    # 單日最大交易量（突發性）
    daily_max = (
        twd_t.assign(date=twd_t["created_at"].dt.date)
        .groupby(["user_id", "date"])
        .size()
        .reset_index(name="daily_cnt")
        .groupby("user_id")["daily_cnt"]
        .max()
        .reset_index(name="twd_max_daily_tx")
    )
    twd_time = twd_time.merge(daily_max, on="user_id", how="left")
    twd_time["twd_burst_ratio"] = (
        twd_time["twd_max_daily_tx"] / twd_time["twd_time_count"].clip(lower=1)
    )

    keep = ["user_id", "twd_active_days", "twd_tx_density",
            "twd_is_short_burst", "twd_is_very_short",
            "twd_max_daily_tx", "twd_burst_ratio"]
    time_feat_parts.append(twd_time[keep])
    print(f"    twd 時間特徵: {len(keep)-1} 個")

# ── 7b. Crypto 時間特徵 ──────────────────────────────────────
if not cryp.empty and "created_at" in cryp.columns:
    cryp_t = cryp.dropna(subset=["created_at"]).copy()

    cryp_time = (
        cryp_t.groupby("user_id")["created_at"]
        .agg(cryp_first_tx="min", cryp_last_tx="max", cryp_time_count="count")
        .reset_index()
    )
    cryp_time["cryp_active_days"] = (
        (cryp_time["cryp_last_tx"] - cryp_time["cryp_first_tx"])
        .dt.total_seconds() / 86400
    ).clip(lower=0)
    cryp_time["cryp_tx_density"]     = cryp_time["cryp_time_count"] / (cryp_time["cryp_active_days"] + 1)
    cryp_time["cryp_is_short_burst"] = (cryp_time["cryp_active_days"] <= 7).astype(int)

    keep = ["user_id", "cryp_active_days", "cryp_tx_density", "cryp_is_short_burst"]
    time_feat_parts.append(cryp_time[keep])
    print(f"    crypto 時間特徵: {len(keep)-1} 個")

# ── 7c. Swap 時間特徵 ────────────────────────────────────────
if not swap.empty and "created_at" in swap.columns:
    swap_t = swap.dropna(subset=["created_at"]).copy()

    swap_time = (
        swap_t.groupby("user_id")["created_at"]
        .agg(swap_first_tx="min", swap_last_tx="max", swap_time_count="count")
        .reset_index()
    )
    swap_time["swap_active_days"] = (
        (swap_time["swap_last_tx"] - swap_time["swap_first_tx"])
        .dt.total_seconds() / 86400
    ).clip(lower=0)
    swap_time["swap_tx_density"]     = swap_time["swap_time_count"] / (swap_time["swap_active_days"] + 1)
    swap_time["swap_is_short_burst"] = (swap_time["swap_active_days"] <= 3).astype(int)

    keep = ["user_id", "swap_active_days", "swap_tx_density", "swap_is_short_burst"]
    time_feat_parts.append(swap_time[keep])
    print(f"    swap 時間特徵: {len(keep)-1} 個")

# ── 7d. 跨管道時間組合特徵 ───────────────────────────────────
if time_feat_parts:
    time_feat = reduce(lambda a, b: a.merge(b, on="user_id", how="outer"), time_feat_parts)
    time_feat = time_feat.fillna(0)

    burst_cols = [c for c in ["twd_is_short_burst", "cryp_is_short_burst", "swap_is_short_burst"]
                  if c in time_feat.columns]
    if burst_cols:
        time_feat["all_channels_burst"] = (
            time_feat[burst_cols].sum(axis=1) == len(burst_cols)
        ).astype(int)
        time_feat["burst_channel_count"] = time_feat[burst_cols].sum(axis=1)

    active_cols = [c for c in ["twd_active_days", "cryp_active_days", "swap_active_days"]
                   if c in time_feat.columns]
    if active_cols:
        # 最短活躍管道的天數（代表「最集中的那個管道」）
        time_feat["min_active_days"] = (
            time_feat[active_cols].replace(0, np.nan).min(axis=1).fillna(0)
        )

    all_feats.append(time_feat)
    n_time_feats = len([c for c in time_feat.columns if c != "user_id"])
    print(f"    時間特徵合計: {n_time_feats} 個")

    print_discrimination(
        time_feat, ui,
        [c for c in time_feat.columns if c != "user_id"]
    )

# ══════════════════════════════════════════════════════════════
# 8. 組合所有特徵
# ══════════════════════════════════════════════════════════════
print("\n  [8] 組合所有特徵...")
feature_matrix = base.copy()
for f in all_feats:
    if not f.empty and "user_id" in f.columns:
        feature_matrix = feature_matrix.merge(f, on="user_id", how="left")

feature_matrix = feature_matrix.fillna(0)

# ══════════════════════════════════════════════════════════════
# 儲存
# ══════════════════════════════════════════════════════════════
out_path = f"{FEAT_DIR}/feature_matrix.parquet"
feature_matrix.to_parquet(out_path, index=False)

feat_cols = [c for c in feature_matrix.columns if c not in ["user_id", "status"]]
print(f"\n✅ 特徵矩陣儲存: {out_path}")
print(f"   shape          : {feature_matrix.shape}")
print(f"   特徵數          : {len(feat_cols)} 個（原本 ~30 個）")
print(f"   新增圖特徵       : graph_* 開頭")
print(f"   新增時間特徵     : *_active_days, *_tx_density, *_burst* 開頭")

print("\n特徵列表:")
for i, c in enumerate(feat_cols, 1):
    tag = " [圖]" if c.startswith("graph_") else " [時間]" if any(
        c.endswith(s) for s in ["_active_days", "_tx_density", "_burst", "_burst_ratio",
                                 "_daily_tx", "burst_channel_count", "min_active_days"]
    ) else ""
    print(f"  {i:3d}. {c}{tag}")

print("\n接著執行: python 04_optimize_v3.py")