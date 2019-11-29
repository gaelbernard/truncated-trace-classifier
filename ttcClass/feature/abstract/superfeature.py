import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
from sklearn.preprocessing import RobustScaler, MinMaxScaler

class FeatureSuperFeature:
    special_end_activity = '$$STOP$$'
    special_start_activity = '$$START$$'
    padding = '$$PADDING$$'

    def __init__(self, input, params):
        self.input = input
        self.params = params
        uactivities = [self.special_end_activity, self.special_start_activity, self.padding] + self.input.df[self.input.activity_column].unique().tolist()
        self.alphabet = {letter:index for index, letter in enumerate(uactivities)}
        self.reverse_alphabet = {index:letter for index, letter in enumerate(uactivities)}
        self.baseFeature = self.build_baseFeature()
        self.data = self.build_feature()

    def build_feature(self):
        exit('Needs to be overwritten by inherited class')
        pass

    def build_baseFeature(self):
        '''
        Extract a set of base feature that will be used by all classifier
        :return:
        '''
        feature = {}

        # Get the cumulative count (i.e., how many activities happened before?)
        feature['cumcount'] = self.input.df.set_index(self.input.case_column).groupby(level=0).cumcount()

        # Get time-related features
        timestamp = self.input.df.set_index(self.input.case_column)[self.input.timestamp_column]
        feature['duration_since_journey_start'] = (timestamp - timestamp.groupby(level=0).min()).dt.total_seconds()
        feature['duration_since_last_event'] = (timestamp - timestamp.groupby(level=0).shift(1)).dt.total_seconds().fillna(0)

        feature = pd.DataFrame(MinMaxScaler().fit_transform(pd.DataFrame(feature)), columns=feature.keys())
        feature.columns = ['BASE$$'+x for x in feature.columns]


        return feature




