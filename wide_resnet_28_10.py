from keras.callbacks import ReduceLROnPlateau#, LearningRateScheduler
from keras.layers import (Input, Conv2D, BatchNormalization, ZeroPadding2D, Dropout,
                          GlobalAveragePooling2D, Activation, Dense, add)
from keras.models import Model
from keras import optimizers
from keras import regularizers
from base_model import BaseModel
from train import train

DEPTH = 28
WIDEN_FACTOR = 10
DROPOUT = 0.3
MODEL_NAME = 'WideResNet28_10' # This should be modified when the model name changes.

class WideResNet28_10(BaseModel):

    def __init__(self):
        # PAPER: Learning rate drops by 0.2 at 60, 120 and 160 epochs. (total 200 epochs)
        #more epochs needed as wideresnet
        callbacks = [ReduceLROnPlateau(monitor = 'val_loss', factor = 0.2,
                                       patience = 10, verbose = 1)]

        self.regularizer = regularizers.l2(5e-04)
        optimizer = optimizers.SGD(lr=0.1, momentum=0.9, decay=1e-04, nesterov = True)
        BaseModel.__init__(self, model = self._build(), optimizer = optimizer,
                           callbacks = callbacks)

    def _build(self):
        '''
        source: 

        - Identity Mappings in Deep Residual Networks (https://arxiv.org/abs/1603.05027)
          => Full pre-activation
          => https://github.com/KaimingHe/resnet-1k-layers/blob/master/resnet-pre-act.lua

        - Wide Residual Networks (https://arxiv.org/abs/1605.07146)
          => No Bottleneck
          => Projection shortcut (B)

        - Author's implementation
          => https://github.com/szagoruyko/wide-residual-networks

        - Global weight decay in keras?
          => https://stackoverflow.com/questions/41260042/global-weight-decay-in-keras

        Returns:
            WideResNet-28-10 model
        '''

        n = (DEPTH - 4) // 6 # Depth should be 6n+4
        k = WIDEN_FACTOR
        nStages = [16, 16 * k, 32 * k, 64 * k]

        x = Input(shape = (28, 28, 1))
        y = ZeroPadding2D(padding = (2, 2))(x) # matching the image size of CIFAR-10

        y = Conv2D(nStages[0], (3, 3), padding = 'same',
                   kernel_regularizer = self.regularizer,
                   bias_regularizer = self.regularizer)(y)
        y = self._layer(y, nStages[1], n, (1, 1)) # spatial size: 32 x 32
        y = self._layer(y, nStages[2], n, (2, 2)) # spatial size: 16 x 16
        y = self._layer(y, nStages[3], n, (2, 2)) # spatial size: 8 x 8
        y = BatchNormalization(beta_regularizer = self.regularizer,
                               gamma_regularizer = self.regularizer)(y)
        y = Activation('relu')(y)
        y = GlobalAveragePooling2D()(y)
        y = Dense(units = 10,
                  kernel_regularizer = self.regularizer,
                  bias_regularizer = self.regularizer)(y)
        y = Activation('softmax')(y)

        return Model(x, y, name = MODEL_NAME)

    def _layer(self, x, output_channel, count, strides):
        '''
        Creates a layer which consists of residual blocks as many as 'count'.

        Returns:
            A layer which consists of multiple residual blocks
        '''
        y = self._wide_basic_residual_block(x, output_channel, True, strides)

        for _ in range(1, count):
            y = self._wide_basic_residual_block(y, output_channel, False, (1, 1))

        return y

    def _wide_basic_residual_block(self, x, output_channel, downsampling, strides):
        '''
        Residual Block: x_{l+1} = x_{l} + F(x_{l}, W_{l})

        Returns:
            a single basic-wide residual block
        '''
        bottleneck_channel = output_channel

        if downsampling:
            # common BN, ReLU
            x = BatchNormalization(beta_regularizer = self.regularizer,
                                   gamma_regularizer = self.regularizer)(x)
            x = Activation('relu')(x)

            fx = Conv2D(bottleneck_channel, (3, 3), padding = 'same', strides = strides,
                        kernel_regularizer = self.regularizer,
                        bias_regularizer = self.regularizer,
                        kernel_initializer = 'he_normal')(x)
        else:
            # conv3x3
            fx = BatchNormalization(beta_regularizer = self.regularizer,
                                    gamma_regularizer = self.regularizer)(x)
            fx = Activation('relu')(fx)
            fx = Conv2D(bottleneck_channel, (3, 3), padding = 'same',
                        kernel_regularizer = self.regularizer,
                        bias_regularizer = self.regularizer,
                        kernel_initializer = 'he_normal')(fx)

        # conv3x3
        fx = BatchNormalization(beta_regularizer = self.regularizer,
                                gamma_regularizer = self.regularizer)(fx)
        fx = Activation('relu')(fx)
        if DROPOUT > 0:
            fx = Dropout(DROPOUT)(fx)
        fx = Conv2D(bottleneck_channel, (3, 3), padding = 'same',
                    kernel_regularizer = self.regularizer,
                    bias_regularizer = self.regularizer,
                    kernel_initializer = 'he_normal')(fx)

        if downsampling:
            # Projection shorcut
            x = Conv2D(output_channel, (1, 1), padding = 'same', strides = strides,
                       kernel_regularizer = self.regularizer,
                       bias_regularizer = self.regularizer,
                       kernel_initializer = 'he_normal')(x)

        return add([x, fx])

def main():
    '''
    Train the model defined above.
    '''
    model = WideResNet28_10()
    train(model, MODEL_NAME)

if __name__ == '__main__':
    main()
