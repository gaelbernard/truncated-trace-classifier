from sklearn.metrics import accuracy_score, recall_score, precision_score, roc_auc_score, roc_curve, f1_score, confusion_matrix, auc, log_loss, matthews_corrcoef
import pandas as pd
import numpy as np
import uuid
import os
from distutils.dir_util import copy_tree

class SuperModel:
    def __init__(self, input, feature, mkdir=True):
        self.id = str(uuid.uuid4())
        self.feature = feature
        self.model = None
        self.input = input
        self.y = self.build_y()
        self.model_directory_path = os.sep.join(['resultsBenchmark', 'models', self.id])
        self.model_file_path = os.sep.join(['resultsBenchmark', 'models', self.id, 'model'])
        self.training_file_path = os.sep.join(['resultsBenchmark', 'models', self.id, 'training'])
        self.prediction_file_path = os.sep.join(['resultsBenchmark', 'models', self.id, 'prediction.csv'])
        if mkdir:
            self.make_directory()

    def build_y(self):
        return (self.input.df[self.input.case_column].shift(-1)==self.input.df[self.input.case_column]).values.astype(bool)

    def build_and_train_model(self, params):
        pass

    def make_prediction(self):
        return self.model.predict(self.feature.data.values[self.input.split.testing_bool,:])

    def evaluate(self, prediction, verbose=False):

        cut_df = self.input.df.loc[self.input.split.testing_bool,:].copy()
        cut_df['never_predicted_complete'] = (prediction).astype(int)

        '''
        !!!! The y label that was used for the training is not the same as the ground truth
        In the training, we made the assumption that the last activity is complete while all the other are incomplete
        This is a correct assumption if we don't have any truncated log in the event logs.
        However, we also manually cut a fraction of the log. Hence, when a log was manually cut,
        the last activity should not be considered as complete.
        This is one the following 3 lines are correcting
        '''
        index_manually_cut = self.input.gt_cut[self.input.gt_cut['cut']==True].index
        ground_truth = pd.Series(self.y[self.input.split.testing_bool], index=self.input.df.loc[self.input.split.testing_bool, self.input.case_column])
        ground_truth.loc[ground_truth.index.isin(index_manually_cut)] = True

        methods = ['accuracy_score', 'recall_score', 'precision_score', 'roc_auc_score', 'roc_curve', 'f1_score', 'confusion_matrix', 'auc', 'log_loss', 'matthews_corrcoef']

        metric = {}
        for method in methods:
            name = 'metric_{}'.format(method)
            to_eval = "{}(prediction, ground_truth)".format(method)
            try:
                metric[name] = eval(to_eval)
            except:
                metric[name] = np.nan

        if verbose:
            print (pd.Series(metric).to_string())

        return metric

    def make_directory(self):
        os.mkdir(self.model_directory_path)

    def export(self, training, prediction):
        self.export_model()
        self.export_training(training)
        self.export_prediction(prediction)

    def export_prediction(self, prediction):
        data = [pd.Series(prediction, name='prediction'), pd.Series(self.y[self.input.split.testing_bool], name='gt')]
        pd.concat(data, axis=1).to_csv(self.prediction_file_path)

    def copy_model(self, id):
        # Copying the model
        copy_tree(os.sep.join(['models', id]), self.model_directory_path)

        # Leave a trace of the original id of the model
        with open(os.sep.join(['models', self.id, 'id_original_model']), 'w') as f:
            f.write(id)

