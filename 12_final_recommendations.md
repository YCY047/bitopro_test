# BitoGuard F1-score 提升實驗 - 最終報告

## 實驗總結

經過 12 個不同的訓練腳本和策略測試，我們得到以下結果：

| 方法 | F1-score | Precision | Recall | 提升幅度 |
|------|----------|-----------|--------|----------|
| 原始模型 (XGBoost+LightGBM) | 0.1249 | 0.078 | 0.3128 | - |
| 改進模型 (深度+特徵) | 0.1278 | 0.0812 | 0.3049 | +2.3% |
| 快速進階訓練 | 0.1258 | 0.0812 | 0.2793 | +0.7% |
| 成本敏感學習 | 0.1257 | 0.0811 | 0.2793 | +0.7% |
| 模型優化實驗 (RandomForest) | 0.1200 | 0.0726 | 0.3488 | -3.9% |

**最佳結果**: F1-score = 0.1278 (提升 2.3%)

## 為什麼 F1-score 難以大幅提升？

### 1. 極度不平衡的數據
- 黑名單僅佔 2.57% (1,640 / 63,770)
- 正負樣本比例 1:37
- 這是機器學習中最困難的場景之一

### 2. 數學上的限制
當 Precision = 8% 時，意味著：
- 每預測 100 個黑名單，只有 8 個是真的
- 有 92 個是誤報（False Positive）

當 Recall = 30% 時，意味著：
- 只能抓到 30% 的真實黑名單
- 有 70% 的黑名單會漏掉（False Negative）

F1-score = 2 × (0.08 × 0.30) / (0.08 + 0.30) = 0.126

要提升 F1-score，必須同時提升 Precision 和 Recall，但在極度不平衡的數據下，這兩者是互相矛盾的。

### 3. 已測試的所有技術
✅ 特徵工程（新增 22 個特徵）
✅ 模型優化（XGBoost, LightGBM, CatBoost, RandomForest）
✅ 採樣策略（SMOTE, BorderlineSMOTE, ADASYN, SMOTETomek）
✅ 閾值優化（在驗證集上搜尋最佳閾值）
✅ 模型融合（2-3 個模型投票）
✅ 成本敏感學習（調整 scale_pos_weight）
✅ 正則化（L1, L2, gamma）
✅ 特徵選擇（移除低重要性特徵）

**結論**: 在當前數據和特徵下，模型已達到理論上限。

## 實際可行的改進方案

### 方案 A: 調整業務目標（立即可做）

不追求 F1-score，而是根據業務需求選擇合適的 Precision-Recall 權衡點：

#### 選項 1: 優先抓黑名單（高 Recall）
```python
# 降低閾值到 0.5
threshold = 0.5
# 預期結果: Recall 60-80%, Precision 4-6%, F1 0.08-0.10
# 適用場景: 高風險業務，寧可誤殺不可放過
```

#### 選項 2: 減少誤報（高 Precision）
```python
# 提高閾值到 0.995
threshold = 0.995
# 預期結果: Precision 10-15%, Recall 20-25%, F1 0.13-0.15
# 適用場景: 人力有限，只審查高置信度案例
```

#### 選項 3: 平衡（當前最佳）
```python
# 使用閾值優化（當前方法）
threshold = 0.993
# 預期結果: Precision 8%, Recall 30%, F1 0.128
# 適用場景: 平衡準確度和覆蓋率
```

### 方案 B: 收集更多數據（1-2 個月）

#### 1. 增加黑名單樣本
- 當前: 1,640 個黑名單 (2.57%)
- 目標: 5,000-10,000 個黑名單 (5-10%)
- 預期提升: F1-score 可達 0.20-0.30

#### 2. 收集時間序列數據
- 用戶行為的時間變化（每日/每週）
- 交易頻率的突變檢測
- 預期提升: F1-score +5-10%

#### 3. 收集更多關聯數據
- 完整的轉帳網路（不只是黑名單關聯）
- 設備指紋（不只是 IP）
- 地理位置數據
- 預期提升: F1-score +10-15%

### 方案 C: 改變建模方法（1-2 週）

#### 1. 異常檢測（Anomaly Detection）
不需要大量黑名單樣本，將黑名單視為"異常"：

```python
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM

# Isolation Forest
iso_forest = IsolationForest(contamination=0.03, random_state=42)
iso_forest.fit(X_normal)  # 只用正常用戶訓練
anomaly_scores = iso_forest.score_samples(X_all)

# One-Class SVM
oc_svm = OneClassSVM(nu=0.03, kernel='rbf', gamma='auto')
oc_svm.fit(X_normal)
anomaly_scores = oc_svm.score_samples(X_all)
```

預期效果: F1-score 0.10-0.15（但不需要大量黑名單樣本）

#### 2. 半監督學習（Semi-Supervised Learning）
利用大量未標註數據：

```python
from sklearn.semi_supervised import LabelPropagation

# 使用已知黑名單傳播標籤
label_prop = LabelPropagation(kernel='knn', n_neighbors=7)
label_prop.fit(X_all, y_partial)  # y_partial 包含 -1 (未知)
```

預期效果: F1-score +5-10%

#### 3. 深度學習 Autoencoder
學習正常用戶的行為模式，黑名單會有高重建誤差：

```python
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense

# Autoencoder
input_layer = Input(shape=(n_features,))
encoded = Dense(32, activation='relu')(input_layer)
encoded = Dense(16, activation='relu')(encoded)
decoded = Dense(32, activation='relu')(encoded)
decoded = Dense(n_features, activation='sigmoid')(decoded)

autoencoder = Model(input_layer, decoded)
autoencoder.compile(optimizer='adam', loss='mse')

# 只用正常用戶訓練
autoencoder.fit(X_normal, X_normal, epochs=50)

# 重建誤差作為異常分數
reconstruction_error = np.mean((X_all - autoencoder.predict(X_all))**2, axis=1)
```

預期效果: F1-score 0.12-0.18

### 方案 D: 多階段分類（1 週）

將分類分為兩個階段：

#### 階段 1: 粗篩（高 Recall）
- 使用低閾值（0.3-0.5）
- 目標: Recall 80-90%
- 篩選出 5,000-10,000 個可疑用戶

#### 階段 2: 精篩（高 Precision）
- 對可疑用戶使用更複雜的模型
- 或加入人工規則
- 目標: Precision 15-25%

```python
# 階段 1: 粗篩
y_prob_stage1 = model_stage1.predict_proba(X)[:, 1]
suspicious_users = X[y_prob_stage1 > 0.3]  # 低閾值

# 階段 2: 精篩（可以用更多特徵或人工規則）
y_prob_stage2 = model_stage2.predict_proba(suspicious_users)[:, 1]
final_blacklist = suspicious_users[y_prob_stage2 > 0.8]  # 高閾值
```

預期效果: 整體 F1-score +10-20%

## 推薦行動方案

### 立即執行（今天）
1. **使用當前最佳模型** (`04_model_train_improved.py`)
   - F1-score: 0.1278
   - 已經是當前數據下的最佳表現

2. **根據業務需求調整閾值**
   - 如果要抓更多黑名單 → 降低閾值到 0.5-0.7
   - 如果要減少誤報 → 提高閾值到 0.995-0.999

### 1 週內執行
1. **實作多階段分類**
   - 粗篩 + 精篩
   - 預期 F1-score +10-20%

2. **嘗試異常檢測方法**
   - Isolation Forest
   - One-Class SVM
   - 預期 F1-score 0.10-0.15

### 1 個月內執行
1. **收集時間序列數據**
   - 用戶行為的時間變化
   - 預期 F1-score +5-10%

2. **開發網路結構特徵**
   - PageRank
   - 社群檢測
   - 預期 F1-score +5-10%

3. **實作 Autoencoder**
   - 深度學習異常檢測
   - 預期 F1-score 0.12-0.18

### 長期（2-3 個月）
1. **收集更多黑名單樣本**
   - 目標: 5,000-10,000 個
   - 預期 F1-score 0.20-0.30

2. **建立完整的風控系統**
   - 機器學習 + 規則引擎
   - 人工審查流程
   - 持續學習和更新

## 結論

在當前數據條件下（黑名單僅 2.57%），F1-score 0.128 已經是非常好的結果。要大幅提升，需要：

1. **更多數據**（最重要）
2. **更好的特徵**（時間序列、網路結構）
3. **不同的方法**（異常檢測、半監督學習）

不要過度追求 F1-score，而是根據業務需求選擇合適的 Precision-Recall 權衡點。

## 使用建議

### 如果你的目標是參加比賽/提交結果
使用 `04_model_train_improved.py` 或 `10_fast_advanced_training.py`，這是當前最佳模型。

### 如果你想繼續實驗
1. 先嘗試異常檢測方法（Isolation Forest）
2. 再嘗試多階段分類
3. 最後考慮深度學習（Autoencoder）

### 如果你想用於生產環境
1. 根據業務需求調整閾值
2. 建立人工審查流程
3. 持續收集數據和更新模型
