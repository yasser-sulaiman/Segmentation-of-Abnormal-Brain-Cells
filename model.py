from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, concatenate, Conv2DTranspose, Cropping2D
from tensorflow.keras.models import Model

def Unet(input_size=(256,256,3)):
    inputs = Input(input_size)
    
    # Contracting path (conv -> conv -> maxpooling)
    # Block1
    conv11 = Conv2D(16, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(inputs)
    conv12 = Conv2D(16, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(conv11)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv12)

    # Block2
    conv21 = Conv2D(32, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(pool1)
    conv22 = Conv2D(32, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(conv21)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv22)

    # Block3
    conv31 = Conv2D(64, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(pool2)
    conv32 = Conv2D(64, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(conv31)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv32)

    # Block4
    conv41 = Conv2D(128, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(pool3)
    conv42 = Conv2D(128, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(conv41)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv42)

    # Block5
    conv51 = Conv2D(256, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(pool4)
    conv52 = Conv2D(256, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(conv51)
    pool5 = MaxPooling2D(pool_size=(2, 2))(conv52)

    # Expansive path (upconv -> concatinate -> conv -> conv)
    # Block6
    up6 = concatenate([Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(conv52), conv42], axis=3)
    conv61 = Conv2D(128, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(up6)
    conv62 = Conv2D(128, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(conv61)

    # Block7
    up7 = concatenate([Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(conv62),conv32], axis=3)
    conv71 = Conv2D(64, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(up7)
    conv72 = Conv2D(64, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(conv71)

    # Block8
    up8 = concatenate([Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(conv72), conv22], axis=3)
    conv81 = Conv2D(32, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(up8)
    conv82 = Conv2D(32, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(conv81)

    # Block9
    up9 = concatenate([Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same')(conv82), conv12], axis=3)
    conv91 = Conv2D(16, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(up9)
    conv92 = Conv2D(16, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(conv91)

    # Final 1x1 convolution
    conv10 = Conv2D(1, (1, 1), activation='sigmoid')(conv92)

    return Model(inputs=[inputs], outputs=[conv10])



if __name__ == '__main__':
    unet = Unet(input_size=(256,256,3))
    unet.summary()
