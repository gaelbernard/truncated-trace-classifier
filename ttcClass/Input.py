import pandas as pd
import numpy as np

class Input:

    def __init__(self, path, noise=0.0, case_column='Case ID', activity_column='Activity', timestamp_column='Complete Timestamp', root='', df=None):
        self.path = path
        self.noise = noise
        self.root_folder = root+'datasets/dataset_for_experiment'
        self.gt_folder = root+'datasets/gt_noise'
        self.case_column = case_column
        self.activity_column = activity_column
        self.timestamp_column = timestamp_column
        if df is None:
            self.df = self.load()
        else:
            self.df = df
        self.split = Split()
        self.split.cut(self.df[self.case_column])
        self.gt_cut = self.load_gt_cut()

    def load(self):
        df = pd.read_csv('{}/cut{}/{}'.format(self.root_folder, self.noise, self.path), parse_dates=[self.timestamp_column], dtype={self.activity_column:str, self.case_column:str}, nrows=None)

        df = df.loc[df[self.case_column].notnull(),:]
        df = df.loc[df[self.activity_column].notnull(),:]
        df = df.loc[df[self.timestamp_column].notnull(),:]
        df.sort_values([self.case_column, self.timestamp_column], inplace=True) #,
        return df

    def load_gt_cut(self):
        df = pd.read_csv('{}/cut{}/{}'.format(self.gt_folder, self.noise, self.path))
        df.columns = ['case', 'cut']
        df = df.set_index('case')
        return df

class Split:
    def __init__(self):
        self.ratio_test = 0.2
        self.training_bool = None
        self.testing_bool = None

    def cut(self, vector_journey_id):

        # We make sure to not cut a journey
        bin1 = int(self.ratio_test*vector_journey_id.nunique())

        # We shuffle the training and validation datasets
        training_and_validation_bool = vector_journey_id.isin(vector_journey_id.unique().tolist()[0:bin1])

        self.training_bool = training_and_validation_bool
        self.testing_bool = vector_journey_id.isin(np.array(vector_journey_id.unique().tolist()[bin1:]))
