import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('reports/1571826958.csv')
df['technique'] = df['feature'] + ' - ' + df['model']

hist = df.groupby(['path','noise','technique'])['metric_matthews_corrcoef'].mean().round(5)

print (hist.to_string())
