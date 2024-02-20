from models.model import Model
from tensorflow.keras import Sequential, layers
from tensorflow.keras.layers.experimental.preprocessing import Rescaling
from tensorflow.keras.optimizers import RMSprop, Adam

class BasicModel(Model):
    def _define_model(self, input_shape, categories_count):
        # Your code goes here
        # you have to initialize self.model to a keras model    
        self.model = Sequential([
            layers.Rescaling(1/255, input_shape=input_shape), #black-white
            layers.Resizing(48,48), 
            layers.Conv2DTranspose(categories_count,(2,2), activation='relu'),
            layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
            layers.Flatten(),
            layers.Dense(512, activation='relu'),
            layers.Activation('relu'),
            layers.BatchNormalization(),
            layers.Dense(categories_count, activation='softmax')
        ])
        # self.model.add(layers.Dense(categories_count, activation='relu', input_shape=input_shape))
        # self.model.add(layers.Dense(categories_count, activation='softmax'))

        # self.model.build(input_shape)
    
    def _compile_model(self):
        # Your code goes here
        # you have to compile the keras model, similar to the example in the writeup
        self.model.compile(
            optimizer=RMSprop(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy'],
        )