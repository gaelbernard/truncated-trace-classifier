import pandas as pd
import os
import numpy as np

case_column = 'Case ID'
activity_column = 'Activity'
timestamp_column = 'Complete Timestamp'
files = ['BPI_12.csv','BPI_13_CP.csv','BPI_13_i.csv','BPI_15_1.csv','BPI_15_2.csv','BPI_15_3.csv','BPI_15_4.csv','BPI_15_5.csv','Env_permit.csv','Helpdesk.csv','BPI_17.csv','BPI_18.csv','BPI_19.csv',]

def cut_log(df, ratio):
    cut = pd.Series(df['Case ID'].unique()).sample(int(df['Case ID'].nunique()*ratio))
    cut = df[df[case_column].isin(cut)].groupby(case_column).count()
    cut_mapping = {_:np.random.randint(0, c.loc[activity_column]) for _, c in cut.iterrows()}
    df.loc[:, 'cut'] = df[case_column].map(cut_mapping).fillna(0)
    df.loc[:, 'cumcount'] = df.groupby(case_column).cumcount(ascending=False)
    df.loc[:, 'to_cut'] = df['cut'] > df['cumcount']
    gt = pd.Series(df[case_column].unique(), index=df[case_column].unique()).isin(cut_mapping.keys()).copy()

    df = df.loc[df['to_cut'] == False,:].copy()
    df.drop(['cut', 'cumcount', 'to_cut'], axis=1, inplace=True)
    return df, gt

for file in files:
    print (file)
    df = pd.read_csv('original_ds/'+file)
    df = df[[case_column, activity_column, timestamp_column]]
    df.sort_values(by=case_column, inplace=True)

    for r in [0.0, 0.10, 0.20]:
        f = 'dataset_for_experiment/cut{}/'.format(r)
        gt_f = 'gt_noise/cut{}/'.format(r)
        if not os.path.exists(f):
            os.makedirs(f)
        if not os.path.exists(gt_f):
            os.makedirs(gt_f)
        df, gt = cut_log(df, ratio=r)
        df.to_csv(f+file)
        gt.to_csv(gt_f+file, header=True)
