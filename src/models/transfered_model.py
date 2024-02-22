from models.model import Model
from tensorflow.keras import Sequential, layers, models
from tensorflow.keras.layers.experimental.preprocessing import Rescaling
from tensorflow.keras.optimizers import RMSprop, Adam

class TransferedModel(Model):
    def _define_model(self, input_shape, categories_count):
        # Your code goes here
        # you have to initialize self.model to a keras model
        # load your basic model with keras's load_model function
        # freeze the weights of the loaded model to make sure the training doesn't affect them
        # (check the number of total params, trainable params and non-trainable params in your summary generated by train_transfer.py)
        # use this model by removing the last layer, adding dense layers and an output layer

        self.model = Model.load_model("step6_submission_basic_model_10_epochs_timestamp_1708452049.keras").model
        for layer in self.model.layers:
            layer.trainable = False

        layer_dense = layers.Dense(256, activation='relu')
        layer_final = layers.Dense(categories_count, activation='softmax')
        self.model.add(layer_dense)
        self.model.add(layer_final)
        print(self.model.layers)

    def _compile_model(self):
        self.model.compile(
            optimizer=RMSprop(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy'],
        )
