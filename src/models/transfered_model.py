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
        pass
    
    def _compile_model(self):
        # Your code goes here
        # you have to compile the keras model, similar to the example in the writeup
        pass