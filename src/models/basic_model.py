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
            layers.Conv2D(categories_count,(2,2), activation='relu'),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Conv2D(64,kernel_size=(3, 3), dilation_rate=(2, 2), activation='relu', padding='same'),
            # layers.DepthwiseConv2D(kernel_size=(3, 3), activation='relu', padding='same'),
            layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
            layers.Conv2D(128,kernel_size=(3, 3), activation='relu', padding='same'),
            layers.MaxPooling2D(),
            layers.SpatialDropout2D(0.1),

            # layers.BatchNormalization(),
            # layers.Dense(32, activation='relu'),
            
            layers.Flatten(),
            layers.Dense(256, activation='relu'),
            # layers.Dense(512, activation='relu'),
            layers.Dropout(0.15),
            # layers.Activation('relu'),
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