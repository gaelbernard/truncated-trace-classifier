from ttcClass.feature.abstract.superfeature import FeatureSuperFeature
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

class FrequencyBased(FeatureSuperFeature):
    def __init__(self, *params):
        super().__init__(*params)

    def build_feature(self):

        # Extracting each prefix
        activities = self.input.df[self.input.activity_column]
        eventLogs = activities.groupby(self.input.df[self.input.case_column]).apply(lambda x: [u for u in x.values.tolist()]).to_list()
        left = [trace[:index] for trace in eventLogs for index in range(1, len(trace)+1)]

        # Occurence count of individual activities
        cv = CountVectorizer(ngram_range=(1,1), tokenizer=lambda doc: doc, binary=False, lowercase=False)
        feature = pd.DataFrame(cv.fit_transform(left).toarray(), columns=cv.get_feature_names())

        # Add the base feature
        feature = pd.concat([feature, self.baseFeature], axis=1)
        return feature
