import pandas as pd

df = pd.read_csv('results.csv')
df['smResultAccuracy'] = df['smResultAccuracy'].round(4)
df['smWithTTCResult'] = df['smWithTTCResult'].round(4)
df['softmax_time_to_build_and_train_model'] = df['softmax_time_to_build_and_train_model'].round(0)
df['ttc_time_to_build_and_train_model'] = df['ttc_time_to_build_and_train_model'].round(0)
df['time_ttc'] = df['softmax_time_to_build_and_train_model']+df['ttc_time_to_build_and_train_model']
df['accuracyIncrease'] = (((df['smWithTTCResult'] - df['smResultAccuracy'])/df['smResultAccuracy'])*100).round(1)
df['timeIncrease'] = ((df['ttc_time_to_build_and_train_model']/(df['time_ttc']))*100).round(1)


df.sort_values('dataset', inplace=True)
print (df.mean())
print (df.to_string())