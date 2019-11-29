from ttcClass.model.abstract.seq import Seq
from keras.layers import Dense, LSTM, Flatten, concatenate, BatchNormalization
from keras.models import Input, Model, load_model
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.initializers import glorot_uniform
from keras import optimizers

class LstmSigmoid(Seq):
    def __init__(self, *params):
        super().__init__(*params)

    def build_and_train_model(self, params):

        input1 = Input(shape=(self.feature.data.shape[1], self.feature.data.shape[2]))
        input2 = Input(shape=(self.feature.baseFeature.shape[1],))
        model1 = LSTM(params['n_cells'], return_sequences=False, dropout=0.5, kernel_initializer=glorot_uniform(0))(input1)
        concat = concatenate([model1, input2])
        concat = Dense(params['n_cells'])(concat)
        out = Dense(1,  activation="sigmoid", kernel_initializer=glorot_uniform(0))(concat) #

        self.model = Model([input1, input2], out) #[input1, input2], out
        self.model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

        es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=10)
        mc = ModelCheckpoint(self.training_file_path+'__best_model.h5', monitor='val_loss', mode='min', verbose=1, save_best_only=True)

        training = self.model.fit(
            [self.feature.data[self.input.split.training_bool.values],self.feature.baseFeature[self.input.split.training_bool.values]],
            self.y[self.input.split.training_bool],
            shuffle=False,
            epochs=params['epoch'],
            verbose=1,
            batch_size=params['batch_size'],
            validation_split=0.2,
            callbacks=[mc, es]
        )
        self.model = load_model(self.training_file_path+'__best_model.h5')

        return training


    def make_prediction(self):
        prediction = self.model.predict(
            [self.feature.data[self.input.split.testing_bool.values], self.feature.baseFeature[self.input.split.testing_bool.values]]
        )
        return prediction.reshape(-1) > .5

