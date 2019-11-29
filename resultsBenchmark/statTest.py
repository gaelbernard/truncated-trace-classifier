import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial.distance import squareform
import seaborn as sns
from matplotlib import pyplot as plt
from mlxtend.evaluate import permutation_test
from scipy.spatial.distance import pdist

# Perform a statistical test of significance on the results
df = pd.read_csv('reports/1571826958.csv')
df['technique'] = df['feature'] + ' - ' + df['model']
hist = df.groupby(['path','noise','technique'])['metric_matthews_corrcoef'].mean().round(5)
hist.index = hist.index.get_level_values(2)

X = []
Y = []
def perm_test(x,y):
    print (x[0], y[0])
    X.append(x[0])
    Y.append(y[0])
    a = hist[hist.index==x[0]].values
    b = hist[hist.index==y[0]].values
    p_value = permutation_test(a, b,
       method='approximate',
       num_rounds=100000,
       seed=0)

    return p_value

u_values = hist.index.unique().values
dist = pdist(u_values.reshape(-1,1), metric=perm_test)
X.extend(Y)
Y.extend(X)
pd.DataFrame([X, Y, list(dist)+list(dist)]).transpose().to_csv('p_val.csv')

