import pandas as pd
import sys
import os

files = os.listdir('logs')
df = pd.DataFrame()
for f in files:
    if f.endswith('.csv'):
        if f == 'exp11_GRU1_alldata_batchsize32_maxlen5_training.csv':
            print('1')
        df1 = pd.read_csv('logs/' + f)
        df1['taskName'] = f.replace('.csv', '')
        new_cols = []
        for col in df1.columns:
            # if col == 'loss':
            #     col = 'all_loss'
            # if col == 'val_loss':
            #     col = 'all_val_loss'
            if 'out_place_' in col:
                newcol = col.replace('out_place_', '')
                if newcol != 'loss' and newcol != 'val_loss':
                    col = newcol
            new_cols.append(col)
        df1.columns = new_cols
        try:
            df = pd.concat([df, df1])
        except Exception as e:
            pass
df.to_csv('allRes.csv', index=False)