"""
BitoGuard - Step 2: 探索性資料分析 (EDA)
執行前需先跑完 01_fetch_data.py
 
使用方式:
    pip install pandas matplotlib seaborn scipy
    python 02_eda.py
"""
 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import warnings
import os
 
warnings.filterwarnings("ignore")
# Windows 中文字體設定
matplotlib.rcParams["font.sans-serif"] = ["Microsoft JhengHei", "Microsoft YaHei", "SimHei"]
matplotlib.rcParams["axes.unicode_minus"] = False  # 解決負號顯示問題
OUTPUT_DIR = "./data/eda_plots"
os.makedirs(OUTPUT_DIR, exist_ok=True)
 
# ── 載入資料 ────────────────────────────────────────────
def load(name):
    path = f"./data/raw/{name}.parquet"
    if not os.path.exists(path):
        print(f"  [SKIP] {path} 不存在，請先執行 01_fetch_data.py")
        return pd.DataFrame()
    return pd.read_parquet(path)
 
print("Loading data ...")
ui   = load("user_info")
twd  = load("twd_transfer")
cryp = load("crypto_transfer")
trade = load("usdt_twd_trading")
swap  = load("usdt_swap")
 
# ── 金額欄位換算（乘以 1e-8）─────────────────────────────
AMOUNT_COLS = {
    "twd_transfer":     ["ori_samount"],
    "crypto_transfer":  ["ori_samount", "twd_srate"],
    "usdt_twd_trading": ["trade_samount", "twd_srate"],
    "usdt_swap":        ["twd_samount", "currency_samount"],
}
for df, cols in [(twd, AMOUNT_COLS["twd_transfer"]),
                 (cryp, AMOUNT_COLS["crypto_transfer"]),
                 (trade, AMOUNT_COLS["usdt_twd_trading"]),
                 (swap, AMOUNT_COLS["usdt_swap"])]:
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce") * 1e-8
 
# datetime 轉型
for df in [twd, cryp, trade, swap]:
    for c in df.columns:
        if "created_at" in c or "updated_at" in c:
            df[c] = pd.to_datetime(df[c], errors="coerce")
 
# ── 1. 黑名單基本分佈 ────────────────────────────────────
print("\n[1] 黑名單分佈")
if not ui.empty and "status" in ui.columns:
    bl_count = ui["status"].value_counts()
    pct = (ui["status"] == 1).mean() * 100
    print(f"  正常用戶: {bl_count.get(0, 0)}")
    print(f"  黑名單:   {bl_count.get(1, 0)}  ({pct:.2f}%)")
 
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.bar(["正常 (0)", "黑名單 (1)"], [bl_count.get(0, 0), bl_count.get(1, 0)],
           color=["steelblue", "crimson"])
    ax.set_title("用戶狀態分佈（不平衡程度）")
    ax.set_ylabel("用戶數")
    for i, v in enumerate([bl_count.get(0, 0), bl_count.get(1, 0)]):
        ax.text(i, v + 5, str(v), ha="center", fontsize=10)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/01_blacklist_dist.png", dpi=150)
    plt.close()
    print("  → 圖表儲存: 01_blacklist_dist.png")
 
# ── 2. 法幣交易行為分析 ──────────────────────────────────
print("\n[2] 法幣（TWD）行為分析")
if not twd.empty and not ui.empty:
    twd_feat = twd.merge(ui[["user_id", "status"]], on="user_id", how="left")
    twd_feat["is_blacklist"] = twd_feat["status"] == 1
 
    # 出入金比率
    twd_ratio = (twd_feat.groupby("user_id")
                 .agg(
                     total=("kind", "count"),
                     withdrawal=("kind", lambda x: (x == 1).sum()),
                     deposit=("kind", lambda x: (x == 0).sum()),
                     is_blacklist=("is_blacklist", "first"),
                 )
                 .assign(withdrawal_ratio=lambda d: d["withdrawal"] / d["total"].clip(lower=1)))
 
    print("  出金比率 (黑名單 vs 正常):")
    print(twd_ratio.groupby("is_blacklist")["withdrawal_ratio"].describe().round(3).to_string())
 
    # 金額分佈
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, label, val in zip(axes, ["正常用戶", "黑名單用戶"], [False, True]):
        data = twd_feat[twd_feat["is_blacklist"] == val]["ori_samount"].dropna()
        ax.hist(data, bins=50, color="steelblue" if not val else "crimson", alpha=0.75)
        ax.set_title(f"法幣金額分佈 — {label}")
        ax.set_xlabel("金額 (TWD)")
        ax.set_ylabel("頻次")
        ax.set_yscale("log")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/02_twd_amount_dist.png", dpi=150)
    plt.close()
    print("  → 圖表儲存: 02_twd_amount_dist.png")
 
# ── 3. IP 跳動分析 ───────────────────────────────────────
print("\n[3] IP 跳動分析")
ip_dfs = []
for df, label in [(twd, "twd"), (cryp, "crypto"), (trade, "trade")]:
    if not df.empty and "source_ip" in df.columns and "user_id" in df.columns:
        tmp = (df.dropna(subset=["source_ip"])
               .groupby("user_id")["source_ip"]
               .nunique()
               .reset_index()
               .rename(columns={"source_ip": f"unique_ip_{label}"}))
        ip_dfs.append(tmp)
 
if ip_dfs:
    from functools import reduce
    ip_feat = reduce(lambda a, b: a.merge(b, on="user_id", how="outer"), ip_dfs)
    ip_feat["total_unique_ip"] = ip_feat.filter(like="unique_ip").sum(axis=1)
    if not ui.empty:
        ip_feat = ip_feat.merge(ui[["user_id", "status"]], on="user_id", how="left")
        print("  IP 跳動 (黑名單 vs 正常):")
        print(ip_feat.groupby("status")["total_unique_ip"].describe().round(2).to_string())
 
        fig, ax = plt.subplots(figsize=(7, 4))
        for s, color, label in [(0, "steelblue", "正常"), (1, "crimson", "黑名單")]:
            d = ip_feat[ip_feat["status"] == s]["total_unique_ip"].dropna()
            ax.hist(d, bins=30, alpha=0.6, color=color, label=label)
        ax.set_title("不同 IP 數量分佈")
        ax.set_xlabel("不同 IP 數量")
        ax.legend()
        plt.tight_layout()
        plt.savefig(f"{OUTPUT_DIR}/03_ip_diversity.png", dpi=150)
        plt.close()
        print("  → 圖表儲存: 03_ip_diversity.png")
 
# ── 4. 深夜交易分析（0~6 點）────────────────────────────
print("\n[4] 深夜交易分析")
night_results = {}
for df, label in [(twd, "twd_transfer"), (trade, "usdt_twd_trading")]:
    if df.empty:
        continue
    time_col = "created_at" if "created_at" in df.columns else "updated_at"
    if time_col not in df.columns:
        continue
    df["hour"] = df[time_col].dt.hour
    df["is_night"] = df["hour"].between(0, 5)
    night = (df.groupby("user_id")
             .agg(total_tx=("hour", "count"),
                  night_tx=("is_night", "sum"))
             .assign(night_ratio=lambda d: d["night_tx"] / d["total_tx"].clip(lower=1)))
    if not ui.empty:
        night = night.merge(ui[["user_id", "status"]], on="user_id", how="left")
        print(f"\n  {label} 深夜比率 (黑名單 vs 正常):")
        print(night.groupby("status")["night_ratio"].describe().round(3).to_string())
    night_results[label] = night
 
# 交易時段熱力圖
if not trade.empty and "hour" in trade.columns and not ui.empty:
    trade_h = trade.merge(ui[["user_id", "status"]], on="user_id", how="left")
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    for ax, s, label in zip(axes, [0, 1], ["正常用戶", "黑名單用戶"]):
        d = trade_h[trade_h["status"] == s]["hour"].value_counts().sort_index()
        ax.bar(d.index, d.values, color="steelblue" if s == 0 else "crimson")
        ax.set_title(f"交易時段分佈 — {label}")
        ax.set_xlabel("小時")
        ax.set_ylabel("交易次數")
        ax.axvspan(-0.5, 5.5, alpha=0.1, color="gray", label="深夜 0~5 時")
        ax.legend()
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/04_hour_dist.png", dpi=150)
    plt.close()
    print("  → 圖表儲存: 04_hour_dist.png")
 
# ── 5. 快進快出滯留時間 ──────────────────────────────────
print("\n[5] 法幣滯留時間分析（入金→出金）")
if not twd.empty:
    tc = "created_at" if "created_at" in twd.columns else None
    if tc:
        dep = twd[twd["kind"] == 0][["user_id", tc]].rename(columns={tc: "deposit_time"})
        wit = twd[twd["kind"] == 1][["user_id", tc]].rename(columns={tc: "withdraw_time"})
        dw = dep.merge(wit, on="user_id")
        dw["stay_hours"] = (dw["withdraw_time"] - dw["deposit_time"]).dt.total_seconds() / 3600
        dw = dw[dw["stay_hours"] > 0]
        dw_min = dw.groupby("user_id")["stay_hours"].min().reset_index(name="min_stay_hours")
        if not ui.empty:
            dw_min = dw_min.merge(ui[["user_id", "status"]], on="user_id", how="left")
            print("  最短滯留時間（小時）:")
            print(dw_min.groupby("status")["min_stay_hours"].describe().round(2).to_string())
 
            fig, ax = plt.subplots(figsize=(7, 4))
            for s, color, label in [(0, "steelblue", "正常"), (1, "crimson", "黑名單")]:
                d = dw_min[dw_min["status"] == s]["min_stay_hours"].clip(upper=72).dropna()
                ax.hist(d, bins=40, alpha=0.6, color=color, label=label)
            ax.set_title("入金→出金 最短滯留時間（小時，截至 72h）")
            ax.set_xlabel("小時")
            ax.legend()
            plt.tight_layout()
            plt.savefig(f"{OUTPUT_DIR}/05_stay_time.png", dpi=150)
            plt.close()
            print("  → 圖表儲存: 05_stay_time.png")
 
# ── 6. KYC 屬性分析 ──────────────────────────────────────
print("\n[6] KYC 屬性分析（職業/收入來源）")
if not ui.empty:
    career_map = {
        1:"農林漁牧",2:"礦業",3:"製造業",4:"電力",5:"用水",6:"營建",
        7:"批發零售",8:"運輸倉儲",9:"住宿餐飲",10:"出版",11:"資訊通訊",
        12:"科技業",13:"金融保險",14:"區塊鏈",15:"不動產",16:"教育",
        17:"醫療",18:"藝文娛樂",19:"服務業",20:"軍公教",21:"公共行政",
        22:"自由業",23:"無業",24:"學生",25:"退休",26:"小企業",
        27:"專業事務所",28:"餐飲業",29:"珠寶銀樓",30:"非營利",31:"彩券"
    }
    ui["career_label"] = ui["career"].map(career_map).fillna("未知")
 
    fig, ax = plt.subplots(figsize=(12, 5))
    bl_career = ui[ui["status"] == 1]["career_label"].value_counts().head(10)
    ax.barh(bl_career.index[::-1], bl_career.values[::-1], color="crimson")
    ax.set_title("黑名單用戶 — 職業分佈（Top 10）")
    ax.set_xlabel("人數")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/06_career_blacklist.png", dpi=150)
    plt.close()
    print("  → 圖表儲存: 06_career_blacklist.png")
 
# ── 7. 相關帳戶內轉分析 ──────────────────────────────────
print("\n[7] 內轉關聯帳戶分析")
if not cryp.empty and "relation_user_id" in cryp.columns:
    bl_ids = set(ui[ui["status"] == 1]["user_id"]) if not ui.empty else set()
    internal = cryp[cryp["sub_kind"] == 1].dropna(subset=["relation_user_id"])
    internal["related_to_blacklist"] = internal["relation_user_id"].isin(bl_ids)
    exposure = (internal.groupby("user_id")["related_to_blacklist"]
                .any().reset_index(name="has_blacklist_contact"))
    total = len(exposure)
    exposed = exposure["has_blacklist_contact"].sum()
    print(f"  與黑名單有直接內轉關係的用戶: {exposed}/{total} ({exposed/total*100:.1f}%)")
    print("  → 此欄位是高價值特徵（圖分析關聯）")
 
# ── 最終摘要 ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("EDA 完成！")
print(f"所有圖表儲存於 {OUTPUT_DIR}/")
print("\n建議重點特徵（依 EDA 結果）：")
print("  1. withdrawal_ratio        — 出金占比")
print("  2. total_unique_ip         — IP 跳動數")
print("  3. night_ratio             — 深夜交易比例")
print("  4. min_stay_hours          — 最短法幣滯留時間")
print("  5. has_blacklist_contact   — 是否與黑名單有內轉")
print("  6. career / income_source  — KYC 屬性")
print("\n接著執行 03_feature_engineering.py")