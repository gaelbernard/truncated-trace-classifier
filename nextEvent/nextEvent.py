import pandas as pd
from ttcClass.Input import Input
from ttcClass.feature.FrequencyBasedandLastActivity import FrequencyBasedandLastActivity
from ttcClass.feature.Sequence3D import Sequence3D
import keras
import joblib
import numpy as np
from sklearn.metrics import accuracy_score

datasets = [
    {'path':'Helpdesk.csv'},
    {'path':'Env_permit.csv'},
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
params = {'n_estimators':200, 'max_depth':8, 'n_cells':16, 'epoch':100, 'batch_size':16}
model = pd.read_csv('../resultsBenchmark/reports/1571826958.csv')

outputs = []
for dataset in datasets:

    # Loading the models (softmax + ttc) generated in the benchmark
    report = model
    report = report[report['path'] == dataset['path']]
    report = report[report['noise'] == 0.0]
    softmaxReport = report[report['model'] == 'LstmSoftmax'].iloc[0].to_dict()
    ttcReport = report[report['feature'] == 'FrequencyBasedandLastActivity'].iloc[0].to_dict()
    ttcModel = joblib.load('../resultsBenchmark/models/'+ttcReport['id']+'/model')
    softmaxModel = keras.models.load_model('../resultsBenchmark/models/'+softmaxReport['id']+'/model')

    # Read input
    input = Input(**dataset, noise=0.0, root='../')

    # Loading feature
    ttcFeature = FrequencyBasedandLastActivity(input, params)
    softmaxFeature = Sequence3D(input, params)
    test = input.df[input.split.testing_bool]
    gt = test[input.activity_column].map(softmaxFeature.alphabet).groupby(test[input.case_column]).shift(-1).fillna(softmaxFeature.alphabet[softmaxFeature.special_end_activity]).astype(int).astype(str)

    # Make prediction with softmax
    smPrediction = np.argmax(softmaxModel.predict([softmaxFeature.data[input.split.testing_bool], softmaxFeature.baseFeature.loc[input.split.testing_bool,:]]), axis=1).astype(int).astype(str)

    # Make prediction with softmax but only if the TTC predict that the trace is truncated
    ttcPrediction = ttcModel.predict(ttcFeature.data.values[input.split.testing_bool,:])
    smWithTTCPrediction = smPrediction.copy()
    smWithTTCPrediction[ttcPrediction == False] = str(ttcFeature.alphabet[ttcFeature.special_end_activity])

    smResult = accuracy_score(gt, smPrediction)
    smWithTTCResult = accuracy_score(gt, smWithTTCPrediction)

    outputs.append({
        'dataset': dataset,
        'smResultAccuracy': smResult,
        'smWithTTCResult': smWithTTCResult,
        'softmax_time_to_build_and_train_model': softmaxReport['time_to_build_and_train_model'],
        'ttc_time_to_build_and_train_model': ttcReport['time_to_build_and_train_model'],
    })

    # Put the results in a report
    print (pd.DataFrame(outputs).to_string())
    print (pd.DataFrame(outputs).to_csv('results.csv'))
