from ttcClass.model.abstract.flat import Flat
import xgboost as xgb
import pandas as pd
from sklearn.model_selection import train_test_split

class FrequencyBased(Flat):
    def __init__(self, *params):
        super().__init__(*params)
        self.end_activity = None

    def build_and_train_model(self, params):

        self.endActivity = self.feature.data['decreasingFactor_cummin'][self.feature.data['decreasingFactor_cummin'] >= params['minDecreasingFactor']].index.values

    def make_prediction(self):
        return ((~self.input.df[self.input.activity_column].isin(self.endActivity)).values[self.input.split.testing_bool])

    def export_training(self, training):
        pass
