from ttcClass.feature.abstract.superfeature import FeatureSuperFeature
import pandas as pd
from keras.utils import to_categorical
from keras.preprocessing.sequence import pad_sequences

class IndexBased(FeatureSuperFeature):
    max_seq_length = 15


    def __init__(self, *params):
        super().__init__(*params)

    def build_feature(self):

        # Get last activity for each prefix length == column activity
        activities = self.input.df[self.input.activity_column]
        activities = activities.map({y:x for x,y in pd.Series(activities.unique()).to_dict().items()})
        sequences = activities.groupby(self.input.df[self.input.case_column]).agg(list).tolist()

        seq = [sequences[case][:i+1] for case, seq in enumerate(sequences) for i in range(len(seq))]

        feature = pd.DataFrame(to_categorical(pad_sequences(seq, maxlen=self.max_seq_length, value=-1)).reshape(len(seq), -1))
        # Add the base feature
        feature = pd.concat([feature, self.baseFeature], axis=1)
        return feature
