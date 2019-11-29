from ttcClass.feature.abstract.superfeature import FeatureSuperFeature
import pandas as pd
from keras.utils import to_categorical

class LastActivity(FeatureSuperFeature):
    def __init__(self, *params):
        super().__init__(*params)

    def build_feature(self):

        # Get last activity for each prefix length == column activity
        activities = self.input.df[self.input.activity_column]

        # One-hot encode and remove activities that never occurs
        feature = pd.DataFrame(to_categorical(activities.map(self.alphabet)))
        feature.columns = [self.reverse_alphabet[x] for x in feature.columns]
        feature = feature.loc[:, feature.sum()>0]

        # Add the base feature
        feature = pd.concat([feature, self.baseFeature], axis=1)
        return feature
