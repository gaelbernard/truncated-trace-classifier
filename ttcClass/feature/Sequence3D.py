from ttcClass.feature.abstract.superfeature import FeatureSuperFeature
from keras.utils import to_categorical, plot_model
from keras.preprocessing.sequence import pad_sequences

class Sequence3D(FeatureSuperFeature):

    max_seq_length = 15

    def __init__(self, *params):
        super().__init__(*params)

    def build_feature(self):

        # Extract sequence of activities (3d np.array)
        activity_vector = self.input.df[self.input.activity_column]
        case_id_vector = self.input.df[self.input.case_column]
        seq_feature = activity_vector.groupby(case_id_vector).apply(lambda x: [self.alphabet[u] for u in x.values.tolist()]).to_list()
        seq_feature = [trace[:index] for trace in seq_feature for index in range(1, len(trace)+1)]
        seq_feature = pad_sequences(seq_feature, maxlen=self.max_seq_length, dtype='int32', value=self.alphabet[self.padding]) #value=len(self.alphabet)
        seq_feature = to_categorical(seq_feature)
        return seq_feature
