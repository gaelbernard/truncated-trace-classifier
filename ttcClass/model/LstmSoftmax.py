from ttcClass.model.abstract.seq import Seq
from keras.layers import Dense, LSTM, Flatten, concatenate, BatchNormalization
from keras.models import Input, Model, load_model
from keras.callbacks import EarlyStopping, ModelCheckpoint
import numpy as np
import pandas as pd
from keras.utils import to_categorical, plot_model
from keras.initializers import glorot_uniform

class LstmSoftmax(Seq):

    def __init__(self, input, mkdir=True):
        super().__init__(input, mkdir)

    def redefine_y(self):
        y = self.input.df[self.input.activity_column].map(self.feature.alphabet).groupby(self.input.df[self.input.case_column]).shift(-1).fillna(self.feature.alphabet[self.feature.special_end_activity]).tolist()
        y = to_categorical(y)
        return y

    def build_and_train_model(self, params):
        target = self.redefine_y()

        input1 = Input(shape=(self.feature.data.shape[1], self.feature.data.shape[2]))
        input2 = Input(shape=(self.feature.baseFeature.shape[1],))
        model1 = LSTM(params['n_cells'], return_sequences=False, dropout=0.5, kernel_initializer=glorot_uniform(0))(input1)
        model1 = BatchNormalization()(model1)
        concat = concatenate([model1, input2])
        concat = Dense(params['n_cells'])(concat)
        out = Dense(target.shape[1],  activation="softmax", kernel_initializer=glorot_uniform(0))(concat)

        self.model = Model([input1, input2], out) #, out
        self.model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

        mc = ModelCheckpoint(self.training_file_path+'__best_model.h5', monitor='val_loss', mode='min', verbose=1, save_best_only=True)
        es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=10)

        training = self.model.fit(
            [self.feature.data[self.input.split.training_bool.values],self.feature.baseFeature[self.input.split.training_bool.values]],
            #[self.feature.data[self.input.split.training_bool.values]],
            target[self.input.split.training_bool],
            shuffle=False,
            epochs=params['epoch'],
            verbose=1,
            batch_size=params['batch_size'],
            validation_split=0.2,
            callbacks=[mc,es]
        )
        self.model = load_model(self.training_file_path+'__best_model.h5')
        return training

    def make_prediction(self):
        prediction = np.argmax(self.model.predict(
            [self.feature.data[self.input.split.testing_bool.values],self.feature.baseFeature[self.input.split.testing_bool.values]]
            #[self.feature.data[self.input.split.testing_bool.values]]
        ), axis=1)

        return prediction != self.feature.alphabet[self.feature.special_end_activity]

