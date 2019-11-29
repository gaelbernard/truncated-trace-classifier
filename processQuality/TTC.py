from sklearn.feature_extraction.text import CountVectorizer
from keras.utils import to_categorical
import xgboost as xgb
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import tree

class TTC:
    def __init__(self, df):
        self.df = df
        self.case_column = 'case:concept:name'
        self.activity_column = 'concept:name'
        self.feature = self.build_feature()
        self.y = self.define_y()

    def build_feature(self):
        # unigram count
        eventLogs = self.df[self.activity_column].groupby(self.df[self.case_column]).apply(lambda x: [u for u in x.values.tolist()]).to_list()
        left = [trace[:index] for trace in eventLogs for index in range(1, len(trace)+1)]
        cv = CountVectorizer(ngram_range=(1,1), tokenizer=lambda doc: doc, binary=False, lowercase=False)
        unigram_count = pd.DataFrame(cv.fit_transform(left).toarray(), columns=cv.get_feature_names())

        # last activity
        alphabet = {x:y for y,x in pd.Series(self.df[self.activity_column].unique()).items()}
        last_activity = pd.DataFrame(to_categorical((self.df[self.activity_column].map(alphabet))))

        # cumcount
        cum_count = self.df[self.activity_column].groupby(self.df[self.case_column]).cumcount()

        return pd.concat([unigram_count.reset_index(), last_activity.reset_index(), cum_count.reset_index()], axis=1).values

    def define_y(self):
        self.df['cumcount'] = self.df.groupby(self.case_column)[self.activity_column].cumcount()
        self.df['max_cumcount'] = self.df[self.case_column].map(self.df.groupby(self.case_column)['cumcount'].max().to_dict())

        return (self.df['cumcount'] != self.df['max_cumcount']).values

    def train(self):
        X_train, X_test, y_train, y_test = train_test_split(self.feature, self.y)

        #self.model = tree.DecisionTreeClassifier(max_depth=3)
        self.model = xgb.XGBClassifier(learning_rate=0.01, n_estimators=1, max_depth=4,
                            min_child_weight=3, gamma=0.0, subsample=0.9, colsample_bytree=1.0,
                            eval_set=[(X_test, y_test)], early_stopping_rounds=10,
                            objective='binary:logistic', nthread=4, eval_metric='auc') #scale_pos_weight=1-self.y.astype(int).mean(),
        self.model.fit(X_train, y_train, ) #eval_metric='auc'

    def predictTruncated(self):
        x = self.feature[self.y != True]
        d = self.model.predict(x)
        case = self.df[self.case_column].unique()

        caseTruncated = case[d==True]
        return caseTruncated