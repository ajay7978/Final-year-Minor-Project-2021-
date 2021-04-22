from keras.callbacks import ReduceLROnPlateau
from keras.layers import DepthwiseConv2D
from keras.layers import (Input, Conv2D, BatchNormalization, ZeroPadding2D,
                          GlobalAveragePooling2D, Activation, Dense)
from keras.models import Model
from keras import optimizers
from base_model import BaseModel
from train import train

ALPHA = 1
MODEL_NAME = f'MobileNet'

class MobileNet(BaseModel):
    def __init__(self):
        callbacks = [ReduceLROnPlateau(monitor = 'val_loss', factor = 0.1,
                                       patience = 30, verbose = 1)]
        optimizer = optimizers.RMSprop(lr = 0.01)
        BaseModel.__init__(self, model = self._build(), optimizer = optimizer,
                           callbacks = callbacks)

    def _build(self):
        '''
        Builds MobileNet.
        - MobileNets (https://arxiv.org/abs/1704.04861)

          => https://github.com/keras-team/keras/blob/master/keras/applications/mobilenet.py
        Returns:
            MobileNet model
        '''
        alpha = ALPHA 
        x = Input(shape = (28, 28, 1))

         # matching the image size of CIFAR-10
        y = ZeroPadding2D(padding = (2, 2))(x)

        y = Conv2D(int(32 * alpha), (3, 3), padding = 'same')(y)
        y = BatchNormalization()(y)
        y = Activation('relu')(y)
        y = self._depthwise_sep_conv(y, 64, alpha) 
        y = self._depthwise_sep_conv(y, 128, alpha, strides = (2, 2)) 
        y = self._depthwise_sep_conv(y, 128, alpha) 
        y = self._depthwise_sep_conv(y, 256, alpha, strides = (2, 2))
        y = self._depthwise_sep_conv(y, 256, alpha) 
        y = self._depthwise_sep_conv(y, 512, alpha, strides = (2, 2)) 
        for _ in range(5):
            y = self._depthwise_sep_conv(y, 512, alpha)
        y = self._depthwise_sep_conv(y, 1024, alpha, strides = (2, 2)) 
        y = self._depthwise_sep_conv(y, 1024, alpha)
        y = GlobalAveragePooling2D()(y)
        y = Dense(units = 10)(y)
        y = Activation('softmax')(y)

        return Model(x, y, name = MODEL_NAME)

    def _depthwise_sep_conv(self, x, filters, alpha, strides = (1, 1)):
       # Creates a depthwise separable convolution block
        y = DepthwiseConv2D((3, 3), padding = 'same', strides = strides)(x)
        y = BatchNormalization()(y)
        y = Activation('relu')(y)
        y = Conv2D(int(filters * alpha), (1, 1), padding = 'same')(y)
        y = BatchNormalization()(y)
        y = Activation('relu')(y)
        return y

def main():
    #Training the model
    model = MobileNet()
    train(model, MODEL_NAME)

if __name__ == '__main__':
    main()
