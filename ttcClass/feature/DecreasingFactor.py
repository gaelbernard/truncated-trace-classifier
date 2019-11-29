from ttcClass.feature.abstract.superfeature import FeatureSuperFeature
import pandas as pd
from keras.utils import to_categorical

class DecreasingFactor(FeatureSuperFeature):
    def __init__(self, *params):
        super().__init__(*params)

    def build_feature(self):
        caseID = self.input.df.loc[self.input.split.training_bool, 'Case ID']
        activity = self.input.df.loc[self.input.split.training_bool, 'Activity']
        vc = activity[caseID != caseID.shift(-1)].value_counts()
        c = pd.DataFrame(vc, index=vc.index)
        c.columns = ['count']
        c['next_count'] = c['count'].shift(1)
        c['decreasingFactor'] = (c['count']/(c['next_count'])).fillna(1)
        c['decreasingFactor_cummin'] = c['decreasingFactor'].cummin()

        return c