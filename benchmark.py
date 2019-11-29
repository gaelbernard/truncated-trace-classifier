import pandas as pd
from time import time

from ttcClass.Input import Input
from ttcClass.model.Xgboost import Xgboost
from ttcClass.model.FrequencyBased import FrequencyBased
from ttcClass.model.LstmSigmoid import LstmSigmoid
from ttcClass.model.LstmSoftmax import LstmSoftmax
from ttcClass.feature.IndexBased import IndexBased
from ttcClass.feature.FrequencyBasedandLastActivity import FrequencyBasedandLastActivity
from ttcClass.feature.LastActivity import LastActivity
from ttcClass.feature.Sequence3D import Sequence3D
from ttcClass.feature.DecreasingFactor import DecreasingFactor

datasets = [
    {'path':'Env_permit.csv'},
    {'path':'Helpdesk.csv'},
    {'path':'BPI_12.csv'},
    {'path':'BPI_13_CP.csv'},
    {'path':'BPI_13_i.csv'},
    {'path':'BPI_15_1.csv'},
    {'path':'BPI_15_2.csv'},
    {'path':'BPI_15_3.csv'},
    {'path':'BPI_15_4.csv'},
    {'path':'BPI_15_5.csv'},
    {'path':'BPI_17.csv'},
    {'path':'BPI_18.csv'},
    {'path':'BPI_19.csv'},
]
import os

# Generating the output folder if not existing
[os.makedirs(f) for f in ['resultsBenchmark', 'resultsBenchmark/reports', 'resultsBenchmark/models'] if not os.path.exists(f)]

report_path = 'resultsBenchmark/reports/{}.csv'.format(int(time()))
report_content = []

configurations = [
    {'model': 'FrequencyBased', 'params': {'minDecreasingFactor': 0.40}, 'feature':'DecreasingFactor'},
    {'model': 'FrequencyBased', 'params': {'minDecreasingFactor': 0.45}, 'feature':'DecreasingFactor'},
    {'model': 'FrequencyBased', 'params': {'minDecreasingFactor': 0.50}, 'feature':'DecreasingFactor'},
    {'model': 'FrequencyBased', 'params': {'minDecreasingFactor': 0.55}, 'feature':'DecreasingFactor'},
    {'model': 'FrequencyBased', 'params': {'minDecreasingFactor': 0.60}, 'feature':'DecreasingFactor'},
    {'model': 'FrequencyBased', 'params': {'minDecreasingFactor': 0.65}, 'feature':'DecreasingFactor'},
    {'model': 'FrequencyBased', 'params': {'minDecreasingFactor': 0.70}, 'feature':'DecreasingFactor'},
    {'model': 'Xgboost', 'params': {'n_estimators':200, 'max_depth':8}, 'feature':'BaseLine'},
    {'model': 'Xgboost', 'params': {'n_estimators':200, 'max_depth':8}, 'feature':'LastActivity'},
    {'model': 'Xgboost', 'params': {'n_estimators':200, 'max_depth':8}, 'feature':'FrequencyBased'},
    {'model': 'Xgboost', 'params': {'n_estimators':200, 'max_depth':8}, 'feature':'FrequencyBasedandLastActivity'},
    {'model': 'LstmSoftmax', 'params': {'n_cells':16, 'epoch':100, 'batch_size':16}, 'feature':'Sequence3D'},
    {'model': 'LstmSigmoid', 'params': {'n_cells':16, 'epoch':100, 'batch_size':16}, 'feature':'Sequence3D'},
]
dfs = {}
for dataset in datasets:
    for noise in [0.0, 0.10, 0.20]:
        for configuration in configurations:
            output = {}

            output.update(dataset)
            output.update(configuration)
            output.update({'noise':noise})

            # Read input
            if dataset['path']+str(noise) in dfs.keys():
                input = Input(**dataset, noise=noise, df=dfs[dataset['path']+str(noise)])
            else:
                input = Input(**dataset, noise=noise)

            dfs[dataset['path']+str(noise)] = input.df

            # Feature
            now = time()
            feature = eval('{}(input, {})'.format(configuration['feature'], configuration['params']))
            output['time_to_build_feature'] = time()-now

            # Building the Model
            model = eval('{}(input, feature)'.format(configuration['model']))
            output['id'] = model.id
            now = time()
            training = model.build_and_train_model(configuration['params'])
            output['time_to_build_and_train_model'] = time()-now
            now = time()

            # Making the prediction
            prediction = model.make_prediction()
            output['time_to_make_prediction'] = time()-now
            score = model.evaluate(prediction)
            output.update(score)

            # Exporting model
            model.export(training, prediction)

            # Saving and showing the results
            report_content.append(output)
            df = pd.DataFrame(report_content)
            display = df[output['path'] == df['path']]
            print (display.to_string())
            df.to_csv(report_path)
