import pandas as pd
 
df = pd.read_parquet('./data/features/feature_matrix.parquet')
bl = df[df['status']==1]
nm = df[df['status']==0]
 
print('=== 黑名單用戶交易活躍度 ===')
print(f'有 swap:        {(bl["swap_tx_count"]>0).sum()} / {len(bl)} ({(bl["swap_tx_count"]>0).mean()*100:.1f}%)')
print(f'有 crypto:      {(bl["crypto_tx_count"]>0).sum()} / {len(bl)} ({(bl["crypto_tx_count"]>0).mean()*100:.1f}%)')
print(f'有 twd:         {(bl["twd_tx_count"]>0).sum()} / {len(bl)} ({(bl["twd_tx_count"]>0).mean()*100:.1f}%)')
print(f'有黑名單聯繫:   {(bl["has_blacklist_contact_1hop"]+bl["has_blacklist_contact_2hop"]>0).sum()} / {len(bl)}')
print(f'完全無交易記錄: {((bl["swap_tx_count"]==0)&(bl["crypto_tx_count"]==0)&(bl["twd_tx_count"]==0)).sum()} / {len(bl)}')
 
print()
print('=== 正常用戶交易活躍度 ===')
print(f'有 swap:        {(nm["swap_tx_count"]>0).sum()} / {len(nm)} ({(nm["swap_tx_count"]>0).mean()*100:.1f}%)')
print(f'有 crypto:      {(nm["crypto_tx_count"]>0).sum()} / {len(nm)} ({(nm["crypto_tx_count"]>0).mean()*100:.1f}%)')
print(f'有 twd:         {(nm["twd_tx_count"]>0).sum()} / {len(nm)} ({(nm["twd_tx_count"]>0).mean()*100:.1f}%)')
print(f'完全無交易記錄: {((nm["swap_tx_count"]==0)&(nm["crypto_tx_count"]==0)&(nm["twd_tx_count"]==0)).sum()} / {len(nm)}')