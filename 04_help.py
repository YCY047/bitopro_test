import pandas as pd
df = pd.read_parquet('./data/features/feature_matrix.parquet')
print([c for c in df.columns if c.startswith('graph_') or 'active_days' in c or 'burst' in c])