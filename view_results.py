"""
在 Terminal 查看模型預測結果
執行方式: python view_results.py
"""

import pandas as pd
import numpy as np

# 載入預測結果
print("=" * 80)
print("BitoGuard 模型預測結果")
print("=" * 80)

# 載入最新的預測結果
pred_df = pd.read_csv("./data/model_output/predictions_improved.csv")

print(f"\n總用戶數: {len(pred_df):,}")
print(f"預測黑名單: {(pred_df['predicted_blacklist']==1).sum():,} 人")
print(f"實際黑名單: {(pred_df['actual_blacklist']==1).sum():,} 人")

# 計算指標
tp = ((pred_df['predicted_blacklist']==1) & (pred_df['actual_blacklist']==1)).sum()
fp = ((pred_df['predicted_blacklist']==1) & (pred_df['actual_blacklist']==0)).sum()
fn = ((pred_df['predicted_blacklist']==0) & (pred_df['actual_blacklist']==1)).sum()
tn = ((pred_df['predicted_blacklist']==0) & (pred_df['actual_blacklist']==0)).sum()

precision = tp / (tp + fp) if (tp + fp) > 0 else 0
recall = tp / (tp + fn) if (tp + fn) > 0 else 0
f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

print(f"\n模型表現:")
print(f"  Precision: {precision:.4f} ({precision*100:.2f}%)")
print(f"  Recall:    {recall:.4f} ({recall*100:.2f}%)")
print(f"  F1-score:  {f1:.4f}")
print(f"\n混淆矩陣:")
print(f"  TP (正確預測黑名單): {tp:,}")
print(f"  FP (誤報):           {fp:,}")
print(f"  FN (漏報):           {fn:,}")
print(f"  TN (正確預測正常):   {tn:,}")

# 顯示高風險用戶 Top 20
print("\n" + "=" * 80)
print("Top 20 高風險用戶")
print("=" * 80)

high_risk = pred_df.nlargest(20, 'risk_score')[['user_id', 'risk_score', 'predicted_blacklist', 'actual_blacklist']]
high_risk['risk_score'] = high_risk['risk_score'].apply(lambda x: f"{x:.2%}")
high_risk['predicted_blacklist'] = high_risk['predicted_blacklist'].map({1: '⚠️ 黑名單', 0: '✅ 正常'})
high_risk['actual_blacklist'] = high_risk['actual_blacklist'].map({1: '🔴 黑名單', 0: '🟢 正常'})
high_risk.columns = ['用戶ID', '風險分數', '預測結果', '實際狀態']

print(high_risk.to_string(index=False))

# 風險分數分佈
print("\n" + "=" * 80)
print("風險分數分佈")
print("=" * 80)

bins = [0, 0.3, 0.5, 0.7, 0.9, 1.0]
labels = ['低風險 (0-30%)', '中低風險 (30-50%)', '中風險 (50-70%)', '高風險 (70-90%)', '極高風險 (90-100%)']
pred_df['risk_level'] = pd.cut(pred_df['risk_score'], bins=bins, labels=labels)

risk_dist = pred_df.groupby('risk_level', observed=True).agg({
    'user_id': 'count',
    'actual_blacklist': 'sum'
}).rename(columns={'user_id': '用戶數', 'actual_blacklist': '實際黑名單數'})

print(risk_dist.to_string())

print("\n" + "=" * 80)
print("完成！")
print("=" * 80)
print("\n提示:")
print("  • 查看完整預測結果: ./data/model_output/predictions_improved.csv")
print("  • 啟動 Dashboard: streamlit run 05_dashboard.py")
print("=" * 80)
