from ttcClass.model.abstract.flat import Flat
import xgboost as xgb
import pandas as pd
from sklearn.model_selection import train_test_split

class Xgboost(Flat):
    def __init__(self, *params):
        super().__init__(*params)

    def build_and_train_model(self, params):

        X_train, X_test, y_train, y_test = train_test_split(self.feature.data.values[self.input.split.training_bool,:], self.y[self.input.split.training_bool], test_size=0.2, random_state=1)

        self.model = xgb.XGBClassifier(learning_rate=0.01, n_estimators=params['n_estimators'], max_depth=params['max_depth'],
                            min_child_weight=3, gamma=0.0, subsample=0.9, colsample_bytree=1.0,
                            eval_set=[(X_test, y_test)], early_stopping_rounds=10,
                            objective='binary:logistic', nthread=4, eval_metric='auc') #scale_pos_weight=1-self.y.astype(int).mean(),

        training = self.model.fit(X_train, y_train, eval_metric='auc')
        return training

