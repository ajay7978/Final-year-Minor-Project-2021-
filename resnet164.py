from keras.callbacks import ReduceLROnPlateau
from keras.layers import (Input, Conv2D, BatchNormalization, ZeroPadding2D,
                          GlobalAveragePooling2D, Activation, Dense, add)
from keras.models import Model
from keras import optimizers
from base_model import BaseModel
from train import train

DEPTH = 164 # or 1001
MODEL_NAME = f'ResNet{DEPTH}'

class ResNet164(BaseModel):
    def __init__(self):
        callbacks = [ReduceLROnPlateau(monitor = 'val_loss', factor = 0.1,
                                       patience = 10, verbose = 1)]
        optimizer = optimizers.SGD(lr=0.1, momentum=0.9, decay=1e-04)
        BaseModel.__init__(self, model = self._build(), optimizer = optimizer,
                           callbacks = callbacks)

    def _build(self):
        '''
        Builds ResNet164.
        - Deep Residual Learning for Image Recognition (https://arxiv.org/abs/1512.03385)
          => Bottleneck
          => Projection shortcut (B)
        - Identity Mappings in Deep Residual Networks (https://arxiv.org/abs/1603.05027)
          => Full pre-activation
        - Author's Implementation
          => https://github.com/KaimingHe/resnet-1k-layers/blob/master/resnet-pre-act.lua

        Returns:
            ResNet164 model
        '''
        n = (DEPTH - 2) // 9
        nStages = [16, 64, 128, 256]

        x = Input(shape = (28, 28, 1))
        y = ZeroPadding2D(padding = (2, 2))(x) # matching the image size of CIFAR-10

        y = Conv2D(nStages[0], (3, 3), padding = 'same')(y)
        y = self._layer(y, nStages[1], n, (1, 1)) # spatial size: 32 x 32
        y = self._layer(y, nStages[2], n, (2, 2)) # spatial size: 16 x 16
        y = self._layer(y, nStages[3], n, (2, 2)) # spatial size: 8 x 8
        y = BatchNormalization()(y)
        y = Activation('relu')(y)
        y = GlobalAveragePooling2D()(y)
        y = Dense(units = 10)(y)
        y = Activation('softmax')(y)

        return Model(x, y, name = MODEL_NAME)

    def _layer(self, x, output_channel, count, strides):
        y = self._residual_block(x, output_channel, True, strides)

        for _ in range(1, count):
            y = self._residual_block(y, output_channel, False, (1, 1))

        return y

    def _residual_block(self, x, output_channel, downsampling, strides):

        bottleneck_channel = output_channel // 4

        if downsampling:

            x = BatchNormalization()(x)
            x = Activation('relu')(x)

            fx = Conv2D(bottleneck_channel, (1, 1), padding = 'same', strides = strides,
                        kernel_initializer = 'he_normal')(x)
        else:

            fx = BatchNormalization()(x)
            fx = Activation('relu')(fx)
            fx = Conv2D(bottleneck_channel, (1, 1), padding = 'same',
                        kernel_initializer = 'he_normal')(fx)

        fx = BatchNormalization()(fx)
        fx = Activation('relu')(fx)
        fx = Conv2D(bottleneck_channel, (3, 3), padding = 'same',
                    kernel_initializer = 'he_normal')(fx)

        # conv1x1
        fx = BatchNormalization()(fx)
        fx = Activation('relu')(fx)
        fx = Conv2D(output_channel, (1, 1), padding = 'same',
                    kernel_initializer = 'he_normal')(fx)

        if downsampling:
            
            x = Conv2D(output_channel, (1, 1), padding = 'same', strides = strides,
                        kernel_initializer = 'he_normal')(x)

        return add([x, fx])

def main():
    '''
    Train the model defined above.
    '''
    model = ResNet164()
    train(model, MODEL_NAME)

if __name__ == '__main__':
    main()
