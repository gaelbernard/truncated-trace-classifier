from ttcClass.model.abstract.supermodel import SuperModel
import matplotlib.pyplot as plt
from keras.utils import to_categorical, plot_model
from keras.models import Input, Model, load_model

class Seq(SuperModel):
    def __init__(self, *params):
        super().__init__(*params)

    def export_training(self, training):
        if 'val_f1' in training.history.keys():
            plt.plot(training.history['f1'])
            plt.plot(training.history['val_f1'])
            plt.title('Model f1')
            plt.ylabel('F1')
            plt.xlabel('Epoch')
            plt.legend(['Train', 'Test'], loc='upper left')
            plt.savefig(self.model_file_path+'_f1.png')
            plt.close()

        # Plot training & validation accuracy values
        plt.plot(training.history['acc'])
        plt.plot(training.history['val_acc'])
        plt.title('Model accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Test'], loc='upper left')
        plt.savefig(self.model_file_path+'_accuracy.png')
        plt.close()

        # Plot training & validation loss values
        plt.plot(training.history['loss'])
        plt.plot(training.history['val_loss'])
        plt.title('Model loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Test'], loc='upper left')
        plt.savefig(self.model_file_path+'_loss.png')
        plt.close()

    def export_model(self):
        self.model.save(self.model_file_path)
        plot_model(self.model, to_file=self.model_file_path+'_network.png', show_shapes=True)

    def load_model(self, id):
        self.copy_model(id)
        self.model = load_model(self.model_file_path)
