import pandas as pd
import numpy as np

from iterstrat.ml_stratifiers import MultilabelStratifiedKFold


df1 = pd.read_csv('cell_train_Riedel_df.csv')
df2 = pd.read_csv('cell_train_df.csv')

df = pd.concat([df1,df2])
labels = [str(i) for i in range(19)]
for x in labels: df[x] = df['image_labels'].apply(lambda r: int(x in r.split('|')))

dfs = df.sample(frac=1, random_state=42)
dfs = dfs.reset_index(drop=True)
len(dfs)

unique_counts = {}
for lbl in labels:
    unique_counts[lbl] = len(dfs[dfs.image_labels == lbl])

full_counts = {}
for lbl in labels:
    count = 0
    for row_label in dfs['image_labels']:
        if lbl in row_label.split('|'): count += 1
    full_counts[lbl] = count
    
counts = list(zip(full_counts.keys(), full_counts.values(), unique_counts.values()))
counts = np.array(sorted(counts, key=lambda x:-x[1]))
counts = pd.DataFrame(counts, columns=['label', 'full_count', 'unique_count'])
counts.set_index('label').T


nfold = 5
#seed = 42

y = dfs[labels].values
X = dfs[['image_id', 'cell_id']].values

dfs['fold'] = np.nan

mskf = MultilabelStratifiedKFold(n_splits=nfold, shuffle=False)
for i, (_, test_index) in enumerate(mskf.split(X, y)):
    dfs.iloc[test_index, -1] = i
    
dfs['fold'] = dfs['fold'].astype('int')

len(dfs)

for i in range(5):
    print(f'Prepared fold: {i}')
    print('Valid size: ', len(dfs[dfs['fold']==i]))
    print('Train size: ', len(dfs[dfs['fold']!=i]))
    dfs[dfs['fold']==i].to_csv(f'valid_fold_{i}.csv', index=False)
    dfs[dfs['fold']!=i].to_csv(f'train_fold_{i}.csv', index=False)