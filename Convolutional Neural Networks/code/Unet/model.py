from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Dropout, Conv2DTranspose, concatenate
from tensorflow.keras.models import Model


def conv_block(inputs=None, n_filters=64, dropout_prob=0, max_pooling=True):
    """
    Conv2D block
    U-Net 모델에서 Conv2D는 2번씩 붙어서 나타난다.
    """
    conv = Conv2D(n_filters, 3, activation='relu', padding='same',
                  kernel_initializer='he_normal')(inputs)
    conv = Conv2D(n_filters, 3, activation='relu', padding='same',
                  kernel_initializer='he_normal')(conv)

    if dropout_prob > 0:
        conv = Dropout(dropout_prob)(conv)

    if max_pooling:
        next_layer = MaxPooling2D(2, strides=2)(conv)

    else:
        next_layer = conv

    skip_connection = conv

    return next_layer, skip_connection


def upsampling_block(expansive_input, contractive_input, n_filters=64):
    """
    upsampling_block
    """
    up = Conv2DTranspose(n_filters, 3, strides=2,
                         padding='same')(expansive_input)
    merge = concatenate([up, contractive_input], axis=3)
    conv = Conv2D(n_filters, 3, activation='relu', padding='same',
                  kernel_initializer='he_normal')(merge)
    conv = Conv2D(n_filters, 3, activation='relu', padding='same',
                  kernel_initializer='he_normal')(conv)

    return conv


def unet_model(input_size=(256, 256, 3), n_filters=64, n_classes=3):
    """
    U-Net 구현
    """
    inputs = Input(input_size)
    cblock1 = conv_block(inputs=inputs, n_filters=n_filters * 1)
    cblock2 = conv_block(inputs=cblock1[0], n_filters=n_filters * 2)
    cblock3 = conv_block(inputs=cblock2[0], n_filters=n_filters * 4)
    cblock4 = conv_block(
        inputs=cblock3[0], n_filters=n_filters * 8, dropout_prob=0.3)

    bottleneck = conv_block(
        inputs=cblock4[0], n_filters=n_filters * 16, dropout_prob=0.3, max_pooling=False)

    ublock6 = upsampling_block(bottleneck[0], cblock4[1], n_filters * 8)
    ublock7 = upsampling_block(ublock6, cblock3[1], n_filters * 4)
    ublock8 = upsampling_block(ublock7, cblock2[1], n_filters * 2)
    ublock9 = upsampling_block(ublock8, cblock1[1], n_filters * 1)

    conv9 = Conv2D(n_filters, 3, activation='relu', padding='same',
                   kernel_initializer='he_normal')(ublock9)
    conv10 = Conv2D(n_classes, 1, padding='same')(conv9)
    model = Model(inputs=inputs, outputs=conv10)
    return model
