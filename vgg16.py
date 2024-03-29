from keras.callbacks import ReduceLROnPlateau
from keras.layers import (Input, Conv2D, BatchNormalization, ZeroPadding2D,
                          MaxPooling2D, Activation, Dense, Dropout, Flatten)
from keras.models import Model
from keras import optimizers
from base_model import BaseModel
from train import train

MODEL_NAME = 'VGG16' # This should be modified when the model name changes.

class VGG16(BaseModel):

    def __init__(self):
        callbacks = [ReduceLROnPlateau(monitor = 'val_loss', factor = 0.1,
                                       patience = 10, verbose = 1)]
        optimizer = optimizers.SGD(lr=0.01, momentum=0.9, decay=1e-04)
        BaseModel.__init__(self, model = self._build(), optimizer = optimizer,
                           callbacks = callbacks)

    def _build(self):
        '''
        Builds VGG16. Details written in the paper below.
        - Very Deep Convolutional Networks for Large-Scale Image Recognition
          (https://arxiv.org/abs/1409.1556)

        Returns:
            VGG16 model
        '''
        x = Input(shape = (28, 28, 1))
        y = ZeroPadding2D(padding = (2, 2))(x) # matching the image size of CIFAR-10

        y = self._multi_conv_pool(y, 64, 2) # 32x32
        y = self._multi_conv_pool(y, 128, 2) # 16x16
        y = self._multi_conv_pool(y, 256, 3) # 8x8
        y = self._multi_conv_pool(y, 512, 3) # 4x4
        y = self._multi_conv_pool(y, 512, 3) # 2x2
        y = Flatten()(y)
        y = Dense(units = 256, activation='relu')(y) # original paper suggests 4096 FC
        y = Dropout(0.5)(y)
        y = Dense(units = 256, activation='relu')(y)
        y = Dropout(0.5)(y)
        y = Dense(units = 10)(y)
        y = Activation('softmax')(y)

        return Model(x, y, name = MODEL_NAME)

    def _multi_conv_pool(self, x, output_channel, n):
        y = x
        for _ in range(n):
            y = Conv2D(output_channel, (3, 3), padding = 'same')(y)
            y = BatchNormalization()(y)
            y = Activation('relu')(y)
        y = MaxPooling2D(strides = (2, 2))(y)
        return y

def main():
    #Train the model.
    model = VGG16()
    train(model, MODEL_NAME)

if __name__ == '__main__':
    main()
