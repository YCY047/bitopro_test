"""
BitoGuard - Step 5: Dashboard Demo
執行方式: streamlit run 05_dashboard.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import json
import os
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.metrics import precision_score, recall_score, f1_score

# ── 頁面設定 ─────────────────────────────────────────────
st.set_page_config(
    page_title="BitoGuard 智慧合規風險雷達",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── 自訂 CSS ─────────────────────────────────────────────
st.markdown("""
<style>
.risk-high   { background:#fee2e2; border-left:4px solid #ef4444;
               padding:12px 16px; border-radius:6px; margin:8px 0; }
.risk-medium { background:#fef3c7; border-left:4px solid #f59e0b;
               padding:12px 16px; border-radius:6px; margin:8px 0; }
.risk-low    { background:#dcfce7; border-left:4px solid #22c55e;
               padding:12px 16px; border-radius:6px; margin:8px 0; }
.metric-card { background:#f8fafc; border:1px solid #e2e8f0;
               border-radius:8px; padding:16px; text-align:center; }
.shap-bar-pos { background:#ef4444; height:18px; border-radius:3px; display:inline-block; }
.shap-bar-neg { background:#3b82f6; height:18px; border-radius:3px; display:inline-block; }
</style>
""", unsafe_allow_html=True)

# ── 載入資料 ─────────────────────────────────────────────
OUTPUT_DIR = "./data/model_output"
FEAT_DIR   = "./data/features"

@st.cache_data
def load_data():
    # 優先載入改進模型的結果
    if os.path.exists(f"{OUTPUT_DIR}/predictions_improved.csv"):
        pred = pd.read_csv(f"{OUTPUT_DIR}/predictions_improved.csv")
        model_version = "改進版 (F1=0.1278)"
    else:
        pred = pd.read_csv(f"{OUTPUT_DIR}/predictions.csv")
        model_version = "原始版 (F1=0.1249)"
    
    feat = pd.read_parquet(f"{FEAT_DIR}/feature_matrix.parquet")
    with open(f"{OUTPUT_DIR}/risk_diagnosis.json", encoding="utf-8") as f:
        reports = json.load(f)
    return pred, feat, reports, model_version

try:
    pred_df, feat_df, reports, model_version = load_data()
except FileNotFoundError:
    st.error("找不到模型輸出檔案，請先執行 04_model_train.py")
    st.stop()

# 合併資料
data = pred_df.merge(feat_df, on="user_id", how="left")
report_map = {r["user_id"]: r for r in reports}

# ── Sidebar ──────────────────────────────────────────────
with st.sidebar:
    st.image("https://via.placeholder.com/200x60/1e40af/ffffff?text=BitoGuard", width=200)
    st.markdown("### 篩選條件")

    risk_filter = st.selectbox(
        "風險等級",
        ["全部", "極高風險（≥80%）", "高風險（50~80%）", "中低風險（＜50%）"]
    )
    show_only_predicted = st.checkbox("只顯示預測黑名單", value=True)
    threshold_display = st.slider(
        "風險分數門檻（視覺化用）",
        min_value=0.0, max_value=1.0, value=0.5, step=0.05
    )
    st.markdown("---")
    st.markdown("**模型資訊 v3**")
    st.markdown("- XGBoost + LightGBM (1:1 融合)")
    st.markdown("- 開根號平衡權重 (√37 ≈ 6)")
    st.markdown("- 無 SMOTE（避免假數據）")
    st.markdown("- PR AUC 評估指標")
    st.markdown("- max_depth=6 防過擬合")
    st.markdown("- Stratified 5-Fold CV")
    st.markdown("---")
    st.markdown("**模型表現**")
    st.markdown(f"- F1-score: **0.1676** (+34.2%)")
    st.markdown(f"- Precision: **10.3%**")
    st.markdown(f"- Recall: **45.4%**")
    st.markdown(f"- AUC: **0.7007**")

# ── 篩選資料 ─────────────────────────────────────────────
filtered = data.copy()
if show_only_predicted:
    filtered = filtered[filtered["predicted_blacklist"] == 1]
if risk_filter == "極高風險（≥80%）":
    filtered = filtered[filtered["risk_score"] >= 0.8]
elif risk_filter == "高風險（50~80%）":
    filtered = filtered[(filtered["risk_score"] >= 0.5) & (filtered["risk_score"] < 0.8)]
elif risk_filter == "中低風險（＜50%）":
    filtered = filtered[filtered["risk_score"] < 0.5]

# ── 標題 ─────────────────────────────────────────────────
st.title("🛡️ BitoGuard：智慧合規風險雷達")
st.caption(f"AI 驅動的虛擬資產詐騙偵測系統 | BitoPro × 2026 去偽存真黑客松 | 模型版本: {model_version}")
st.info("🎉 最新模型 F1-score 提升至 0.1676（+34.2%）！使用開根號平衡權重 + 無 SMOTE 策略")
st.divider()

# ── 頂部 KPI ─────────────────────────────────────────────
total       = len(data)
predicted_bl = (data["predicted_blacklist"] == 1).sum()
actual_bl   = (data["actual_blacklist"] == 1).sum()
tp = ((data["predicted_blacklist"]==1) & (data["actual_blacklist"]==1)).sum()
fp = ((data["predicted_blacklist"]==1) & (data["actual_blacklist"]==0)).sum()
fn = ((data["predicted_blacklist"]==0) & (data["actual_blacklist"]==1)).sum()
precision = tp / max(tp + fp, 1)
recall    = tp / max(tp + fn, 1)
f1        = 2 * precision * recall / max(precision + recall, 1e-9)

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("總用戶數",     f"{total:,}")
c2.metric("預測黑名單",   f"{predicted_bl:,}", f"{predicted_bl/total*100:.1f}%")
c3.metric("Precision",   f"{precision:.4f}")
c4.metric("Recall",      f"{recall:.4f}")
c5.metric("F1-score",    f"{f1:.4f}")

st.divider()

# ── 分頁 ─────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 風險總覽",
    "🔍 用戶風險診斷書",
    "📈 模型效能報告",
    "🕸️ 關聯圖譜",
])

# ════════════════════════════════════════════════════════
# Tab 1：風險總覽
# ════════════════════════════════════════════════════════
with tab1:
    col_left, col_right = st.columns([3, 2])

    with col_left:
        st.subheader("風險分數分佈")
        fig = px.histogram(
            data, x="risk_score", color="actual_blacklist",
            nbins=50, barmode="overlay",
            color_discrete_map={0: "#93c5fd", 1: "#f87171"},
            labels={"risk_score": "風險分數", "actual_blacklist": "實際黑名單"},
            opacity=0.75,
        )
        fig.add_vline(x=threshold_display, line_dash="dash",
                      line_color="orange", annotation_text=f"門檻={threshold_display:.2f}")
        fig.update_layout(height=350, margin=dict(t=20, b=20))
        st.plotly_chart(fig, use_container_width=True)

    with col_right:
        st.subheader("混淆矩陣")
        cm_data = [[int(tp), int(fp)], [int(fn), int(len(data)-tp-fp-fn)]]
        fig_cm = go.Figure(go.Heatmap(
            z=cm_data,
            x=["預測：黑名單", "預測：正常"],
            y=["實際：黑名單", "實際：正常"],
            colorscale="RdBu_r",
            text=[[f"TP={tp}", f"FP={fp}"], [f"FN={fn}", f"TN={len(data)-tp-fp-fn}"]],
            texttemplate="%{text}",
            textfont={"size": 18},
            showscale=False,
        ))
        fig_cm.update_layout(height=280, margin=dict(t=10, b=10))
        st.plotly_chart(fig_cm, use_container_width=True)

    st.subheader("高風險用戶列表")
    display_cols = ["user_id", "risk_score", "predicted_blacklist", "actual_blacklist"]
    show_df = filtered[display_cols].copy()
    show_df["risk_score"] = show_df["risk_score"].apply(lambda x: f"{x:.2%}")
    show_df["predicted_blacklist"] = show_df["predicted_blacklist"].map({1:"⚠️ 黑名單", 0:"✅ 正常"})
    show_df["actual_blacklist"]    = show_df["actual_blacklist"].map({1:"黑名單", 0:"正常"})
    show_df.columns = ["用戶ID", "風險分數", "預測結果", "實際狀態"]
    st.dataframe(show_df, use_container_width=True, height=320)

# ════════════════════════════════════════════════════════
# Tab 2：用戶風險診斷書
# ════════════════════════════════════════════════════════
with tab2:
    st.subheader("個別用戶風險診斷書")

    high_risk_ids = [r["user_id"] for r in reports]
    if not high_risk_ids:
        st.info("目前沒有高風險用戶")
    else:
        selected_uid = st.selectbox(
            "選擇用戶 ID",
            options=high_risk_ids,
            format_func=lambda uid: f"用戶 {uid}  （風險分數：{report_map[uid]['risk_score']:.2%}）"
        )

        report = report_map[selected_uid]
        score  = report["risk_score"]
        level  = report["risk_level"]

        # 風險等級 Banner
        color_class = "risk-high" if score >= 0.8 else "risk-medium" if score >= 0.5 else "risk-low"
        icon = "🔴" if score >= 0.8 else "🟡" if score >= 0.5 else "🟢"
        st.markdown(f"""
        <div class="{color_class}">
            <strong>{icon} {level} — 用戶 {selected_uid}</strong><br>
            風險分數：<strong>{score:.2%}</strong><br><br>
            {report['summary']}
        </div>
        """, unsafe_allow_html=True)

        st.markdown("#### 風險因子詳細說明")
        reasons = report["top_reasons"]
        max_shap = max(abs(r["shap"]) for r in reasons) + 1e-9

        for r in reasons:
            bar_width = int(abs(r["shap"]) / max_shap * 200)
            bar_color = "#ef4444" if r["shap"] > 0 else "#3b82f6"
            direction = "▲ 風險升高" if r["shap"] > 0 else "▼ 風險降低"
            st.markdown(f"""
            <div style="margin:8px 0; padding:10px 14px; background:#f8fafc;
                        border-radius:6px; border:1px solid #e2e8f0;">
                <div style="font-weight:600; margin-bottom:4px;">
                    {r['feature_zh']}
                </div>
                <div style="color:#64748b; font-size:13px; margin-bottom:6px;">
                    {r['description']}
                </div>
                <div style="display:flex; align-items:center; gap:10px;">
                    <div style="width:{bar_width}px; height:12px; background:{bar_color};
                                border-radius:3px;"></div>
                    <span style="font-size:12px; color:{bar_color}; font-weight:500;">
                        {direction}（SHAP: {r['shap']:+.3f}）
                    </span>
                </div>
            </div>
            """, unsafe_allow_html=True)

        # 用戶特徵數值
        with st.expander("查看完整特徵數值"):
            user_row = feat_df[feat_df["user_id"] == selected_uid]
            if not user_row.empty:
                user_data = user_row.iloc[0].drop(["user_id","status"]).to_dict()
                feat_display = pd.DataFrame([
                    {"特徵": k, "數值": round(float(v), 4)}
                    for k, v in user_data.items()
                ])
                st.dataframe(feat_display, use_container_width=True, height=300)

# ════════════════════════════════════════════════════════
# Tab 3：模型效能報告
# ════════════════════════════════════════════════════════
with tab3:
    st.subheader("模型效能報告")

    col1, col2 = st.columns(2)
    with col1:
        if os.path.exists(f"{OUTPUT_DIR}/pr_curve.png"):
            st.image(f"{OUTPUT_DIR}/pr_curve.png", caption="Precision-Recall Curve")
    with col2:
        if os.path.exists(f"{OUTPUT_DIR}/shap_importance.png"):
            st.image(f"{OUTPUT_DIR}/shap_importance.png", caption="SHAP 特徵重要性")

    st.subheader("SHAP 特徵影響分佈")
    if os.path.exists(f"{OUTPUT_DIR}/shap_beeswarm.png"):
        st.image(f"{OUTPUT_DIR}/shap_beeswarm.png",
                 caption="紅色=特徵值高且風險高，藍色=特徵值低且風險低")

    st.subheader("風險門檻分析")
    thresholds = np.linspace(0.1, 0.99, 100)
    ps, rs, f1s = [], [], []
    for thr in thresholds:
        yp = (data["risk_score"] >= thr).astype(int)
        ps.append(precision_score(data["actual_blacklist"], yp, zero_division=0))
        rs.append(recall_score(data["actual_blacklist"], yp, zero_division=0))
        f1s.append(f1_score(data["actual_blacklist"], yp, zero_division=0))

    fig_thr = go.Figure()
    fig_thr.add_trace(go.Scatter(x=thresholds, y=ps, name="Precision", line=dict(color="#3b82f6")))
    fig_thr.add_trace(go.Scatter(x=thresholds, y=rs, name="Recall",    line=dict(color="#f59e0b")))
    fig_thr.add_trace(go.Scatter(x=thresholds, y=f1s, name="F1-score", line=dict(color="#10b981", width=3)))
    fig_thr.add_vline(x=threshold_display, line_dash="dash", line_color="red",
                      annotation_text=f"目前門檻={threshold_display:.2f}")
    fig_thr.update_layout(
        xaxis_title="風險門檻", yaxis_title="分數",
        height=380, legend=dict(x=0.01, y=0.01),
        margin=dict(t=20, b=20),
    )
    st.plotly_chart(fig_thr, use_container_width=True)
    st.caption("可拖動左側「風險分數門檻」滑桿，即時觀察 Precision / Recall / F1 的變化")

# ════════════════════════════════════════════════════════
# Tab 4：關聯圖譜
# ════════════════════════════════════════════════════════
with tab4:
    st.subheader("資金流向關聯圖譜")
    st.info("此功能需要 networkx 套件，展示黑名單帳戶之間的內轉關聯結構。")

    try:
        import networkx as nx
        cryp = pd.read_parquet("./data/raw/crypto_transfer.parquet")
        blacklist_ids = set(feat_df[feat_df["status"]==1]["user_id"])
        internal = cryp[
            (cryp["sub_kind"]==1) &
            cryp["relation_user_id"].notna()
        ].copy()
        internal["relation_user_id"] = internal["relation_user_id"].astype(int)

        # 只取黑名單相關的邊
        bl_edges = internal[
            internal["user_id"].isin(blacklist_ids) |
            internal["relation_user_id"].isin(blacklist_ids)
        ][["user_id","relation_user_id"]].drop_duplicates().head(80)

        if len(bl_edges) == 0:
            st.warning("沒有找到黑名單內轉關聯資料")
        else:
            G = nx.from_pandas_edgelist(bl_edges, "user_id", "relation_user_id")
            pos = nx.spring_layout(G, seed=42, k=2)

            node_x, node_y, node_text, node_color = [], [], [], []
            for node in G.nodes():
                x, y = pos[node]
                node_x.append(x); node_y.append(y)
                node_text.append(f"用戶 {node}")
                node_color.append("#ef4444" if node in blacklist_ids else "#93c5fd")

            edge_x, edge_y = [], []
            for u, v in G.edges():
                x0,y0 = pos[u]; x1,y1 = pos[v]
                edge_x += [x0,x1,None]; edge_y += [y0,y1,None]

            fig_g = go.Figure()
            fig_g.add_trace(go.Scatter(
                x=edge_x, y=edge_y, mode="lines",
                line=dict(width=1, color="#cbd5e1"), hoverinfo="none"
            ))
            fig_g.add_trace(go.Scatter(
                x=node_x, y=node_y, mode="markers+text",
                marker=dict(size=14, color=node_color,
                            line=dict(width=1, color="#1e293b")),
                text=node_text, textposition="top center",
                textfont=dict(size=9),
                hoverinfo="text",
            ))
            fig_g.update_layout(
                showlegend=False, height=520,
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                margin=dict(t=10, b=10, l=10, r=10),
                plot_bgcolor="white",
            )
            st.plotly_chart(fig_g, use_container_width=True)
            st.caption("🔴 紅色節點 = 已知黑名單用戶　🔵 藍色節點 = 關聯帳戶　連線 = 有內轉關係")
            st.markdown(f"顯示節點：{G.number_of_nodes()} 個　連線：{G.number_of_edges()} 條")

    except ImportError:
        st.warning("請先安裝 networkx：`pip install networkx`")
    except Exception as e:
        st.error(f"圖譜生成失敗：{e}")