from ttcClass.model.abstract.supermodel import SuperModel
from joblib import dump, load
import pandas as pd
import os

class Flat(SuperModel):
    def __init__(self, *params):
        super().__init__(*params)

    def export_model(self):
        dump(self.model, self.model_file_path)

    def export_training(self, training):
        dump(training, self.training_file_path)
        path = os.sep.join(['resultsBenchmark', 'models', self.id, 'feature_importance.csv'])
        pd.Series(self.model.feature_importances_, index=self.feature.data.columns).sort_values(ascending=False).to_csv(path, header=True)

    def load_model(self, id):
        self.copy_model(id)
        self.model = load(self.model_file_path)

