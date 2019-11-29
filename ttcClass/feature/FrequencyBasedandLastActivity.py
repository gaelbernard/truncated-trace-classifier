from ttcClass.feature.abstract.superfeature import FeatureSuperFeature
import pandas as pd
from keras.utils import to_categorical, plot_model
from sklearn.feature_extraction.text import CountVectorizer

class FrequencyBasedandLastActivity(FeatureSuperFeature):
    def __init__(self, *params):
        super().__init__(*params)

    def build_feature(self):

        # Extracting each prefix
        activities = self.input.df[self.input.activity_column]
        eventLogs = activities.groupby(self.input.df[self.input.case_column]).apply(lambda x: [u for u in x.values.tolist()]).to_list()
        left = [trace[:index] for trace in eventLogs for index in range(1, len(trace)+1)]

        # Occurence count of individual activities
        cv = CountVectorizer(ngram_range=(1,1), tokenizer=lambda doc: doc, binary=False, lowercase=False)
        occurence_feature = pd.DataFrame(cv.fit_transform(left).toarray(), columns=cv.get_feature_names())

        # Get last activity for each prefix length == column activity
        activities = self.input.df[self.input.activity_column]

        # One-hot encode and remove activities that never occurs
        last_activity_feature = pd.DataFrame(to_categorical(activities.map(self.alphabet)))
        last_activity_feature.columns = [self.reverse_alphabet[x] for x in last_activity_feature.columns]
        last_activity_feature = last_activity_feature.loc[:, last_activity_feature.sum()>0]

        # Add the base feature
        feature = pd.concat([occurence_feature, last_activity_feature, self.baseFeature], axis=1)
        #feature = pd.concat([occurence_feature, last_activity_feature], axis=1)
        return feature
