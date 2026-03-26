import pandas as pd
import os
from sklearn.metrics import f1_score, precision_score, recall_score

results = []

pred_files = [
    "predictions.csv",
    "predictions_improved.csv", 
    "predictions_v3.csv",
    "predictions_stacking.csv",
    "predictions_advanced.csv",
    "predictions_cost_sensitive.csv",
]

for file in pred_files:
    path = f"./data/model_output/{file}"
    if os.path.exists(path):
        df = pd.read_csv(path)
        
        # 找出預測和實際列名
        pred_cols = [c for c in df.columns if 'predicted' in c.lower()]
        actual_cols = [c for c in df.columns if 'actual' in c.lower() or 'status' in c.lower()]
        
        if not pred_cols or not actual_cols:
            print(f"Skipping {file}: missing columns")
            continue
            
        pred_col = pred_cols[0]
        actual_col = actual_cols[0]
        
        y_pred = df[pred_col]
        y_true = df[actual_col]
        
        f1 = f1_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)
        
        results.append({
            'File': file,
            'F1-score': f1,
            'Precision': precision,
            'Recall': recall
        })

results_df = pd.DataFrame(results).sort_values('F1-score', ascending=False)

print("=" * 80)
print("所有模型 F1-score 比較")
print("=" * 80)
print(results_df.to_string(index=False))
print("=" * 80)
print(f"\n🏆 最高 F1-score: {results_df.iloc[0]['F1-score']:.4f}")
print(f"   來自: {results_df.iloc[0]['File']}")
print(f"   Precision: {results_df.iloc[0]['Precision']:.4f}")
print(f"   Recall: {results_df.iloc[0]['Recall']:.4f}")
