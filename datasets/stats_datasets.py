import pandas as pd
files = ['BPI_19.csv', 'BPI_12.csv','BPI_13_CP.csv','BPI_13_i.csv','BPI_15_1.csv','BPI_15_2.csv','BPI_15_3.csv','BPI_15_4.csv','BPI_15_5.csv','BPI_17.csv','BPI_18.csv','Env_permit.csv','Helpdesk.csv']


#print ('nunique line', df.shape[0])
#print ('nunique case', df[case_column].nunique())
#print ('avg length', df.shape[0]/df[case_column].nunique())
#print ('nunique activity', df[activity_column].nunique())
#print ('longest case', df.groupby(case_column)[case_column].count().max())
case_column='Case ID'
activity_column='Activity'
timestamp_column='Complete Timestamp'
sigma = '\u03C3'

stat_ds = []
for dataset in files:
    df = pd.read_csv('dataset_for_experiment/cut0.0/'+dataset)
    stat = {}
    stat['#activities'] = str(round(df.shape[0] / 1000,1))+'K'
    stat['#activities_num'] = df.shape[0]
    stat['#'+sigma] = str(round(df[case_column].nunique() / 1000,1))+'K'
    stat[sigma+'_num'] = round(df[case_column].nunique(),1)
    stat[sigma+' mean length'] = round(df.shape[0]/df[case_column].nunique(),1)
    stat[sigma+' max length'] = df.groupby(case_column)[case_column].count().max()
    stat[sigma+' min length'] = df.groupby(case_column)[case_column].count().min()
    stat['|A|'] = df[activity_column].nunique()
    stat['dataset'] = dataset
    stat_ds.append(stat)

stat_ds = pd.DataFrame(stat_ds).set_index('dataset')
stat_ds.to_csv('stat.csv')
print (stat_ds.to_string())


