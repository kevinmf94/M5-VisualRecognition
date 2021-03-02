
from keras.engine.input_layer import Input
from keras.layers.convolutional import Convolution2D, ZeroPadding2D
from keras.layers.pooling import AveragePooling2D, MaxPooling2D
from keras.layers.core import Activation, Dense, Dropout, Flatten
from keras.layers.merge import Concatenate
from keras.engine.training import Model
from keras.layers.normalization import BatchNormalization
import keras

def LeNet5(img_width, img_height):

    input = Input(shape=(img_width, img_height, 3))
    x = Convolution2D(6, kernel_size=5, activation='relu')(input)
    x = MaxPooling2D(pool_size=(2, 2), strides=2)(x)
    x = Convolution2D(16, kernel_size=5, activation='relu')(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=2)(x)
    x = Flatten()(x)
    x = Dense(120, activation='relu')(x)
    x = Dense(84, activation='relu')(x)
    x = Dense(8, activation='softmax')(x)

    return Model(input, x, name="LeNet5")

def LeNet5Conv2D(img_width, img_height):
    
    input = Input(shape=(img_width, img_height, 3))
    x = Convolution2D(6, kernel_size=5, activation='relu')(input)
    x = MaxPooling2D(pool_size=(2, 2), strides=2)(x)
    x = Convolution2D(16, kernel_size=5, activation='relu')(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=2)(x)
    x = Convolution2D(32, kernel_size=5, activation='relu')(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=2)(x)
    x = Flatten()(x)
    x = Dense(120, activation='relu')(x)
    x = Dense(84, activation='relu')(x)
    x = Dense(8, activation='softmax')(x)

    return Model(input, x, name="LeNet5")

def Model3(img_width, img_height):
    
    input = Input(shape=(img_width, img_height, 3))

    x = Convolution2D(10, kernel_size=1, activation='relu')(input)
    x = MaxPooling2D(pool_size=2, strides=2)(x)
    x = Convolution2D(16, kernel_size=3, strides=1, activation='relu')(x)
    x = MaxPooling2D(pool_size=2, strides=2)(x)
    
    x = Flatten()(x)
    x = Dropout(0.3)(x)
    x = Dense(256, activation='relu')(x)
    x = Dense(8, activation='softmax')(x)

    return Model(input, x, name="Model3")

def Model4(img_width, img_height):
    
    input = Input(shape=(img_width, img_height, 3))

    x = Convolution2D(64, kernel_size=1, activation='relu')(input)
    x = MaxPooling2D(pool_size=2, strides=2)(x)
    x = Convolution2D(128, kernel_size=3, strides=1, activation='relu')(x)
    x = MaxPooling2D(pool_size=2, strides=2)(x)
    
    x = Flatten()(x)
    x = Dropout(0.4)(x)
    x = Dense(256, activation='relu')(x)
    x = Dense(8, activation='softmax')(x)

    return Model(input, x, name="Model4")

def Model6(img_width, img_height):
    
    input = Input(shape=(img_width, img_height, 3))

    x = Convolution2D(8, kernel_size=3, activation='relu')(input)
    x = MaxPooling2D(pool_size=2, strides=2)(x)
    x = Convolution2D(16, kernel_size=1, strides=1, activation='relu')(x)
    x = MaxPooling2D(pool_size=2, strides=2)(x)
    x = Convolution2D(32, kernel_size=1, strides=1, activation='relu')(x)
    x = MaxPooling2D(pool_size=2, strides=2)(x)
    
    x = Flatten()(x)
    x = Dropout(0.4)(x)
    x = Dense(1000, activation='relu')(x)
    x = Dense(8, activation='softmax')(x)

    return Model(input, x, name="Model6")


def AlexNet(img_width, img_height):

    input = Input(shape=(img_width, img_height, 3))

    x = Convolution2D(96, kernel_size=11, strides=4, activation='relu')(input)
    x = MaxPooling2D(3, strides=2)(x)
    x = Convolution2D(256, kernel_size=5, padding='same', activation='relu')(x)
    x = MaxPooling2D(3, strides=2)(x)
    x = Convolution2D(384, kernel_size=3, padding='same', activation='relu')(x)
    x = Convolution2D(384, kernel_size=3, padding='same', activation='relu')(x)
    x = Convolution2D(256, kernel_size=3, padding='same', activation='relu')(x)
    x = MaxPooling2D(3, strides=2)(x)
    x = Flatten()(x)

    x = Dense(4096, activation='relu')(x)
    x = Dense(4096, activation='relu')(x)
    x = Dense(8, activation='softmax')(x)

    return Model(input, x, name="AlexNet")


def Model7(img_width, img_height):
    
    input = Input(shape=(img_width, img_height, 3))

    x = Convolution2D(4, kernel_size=5, activation='relu')(input)
    x = MaxPooling2D(2, strides=2)(x)
    x = Convolution2D(10, kernel_size=5, activation='relu')(x)
    x = MaxPooling2D(2, strides=2)(x)
    x = Flatten()(x)

    x = Dense(128, activation='relu')(x)
    x = Dense(64, activation='relu')(x)
    x = Dense(8, activation='softmax')(x)

    return Model(input, x, name="Model7")

def Model7V2(img_width, img_height):
    
    input = Input(shape=(img_width, img_height, 3))

    x = Convolution2D(4, kernel_size=5, activation='relu')(input)
    x = MaxPooling2D(2, strides=2)(x)
    x = Convolution2D(12, kernel_size=5, activation='relu')(x)
    x = MaxPooling2D(2, strides=2)(x)
    x = Flatten()(x)

    x = Dense(120, activation='relu')(x)
    x = Dense(84, activation='relu')(x)
    x = Dense(8, activation='softmax')(x)

    return Model(input, x, name="Model7V2")

def Model8(img_width, img_height):
    
    input = Input(shape=(img_width, img_height, 3))

    input1 = Dropout(0.1)(input)

    x = Convolution2D(6, kernel_size=5, activation='relu')(input1)
    x = MaxPooling2D(2, strides=2)(x)
    x = Convolution2D(16, kernel_size=5, activation='relu')(x)
    x = MaxPooling2D(2, strides=2)(x)
    x = Flatten()(x)
    
    y = Convolution2D(6, kernel_size=5, activation='relu')(input1)
    y = MaxPooling2D(2, strides=2)(y)
    y = Convolution2D(6, kernel_size=5, activation='relu')(y)
    y = MaxPooling2D(2, strides=2)(y)
    y = Flatten()(y)

    z = Convolution2D(6, kernel_size=5, activation='relu')(input1)
    z = MaxPooling2D(2, strides=2)(z)
    z = Convolution2D(16, kernel_size=5, activation='relu')(z)
    z = MaxPooling2D(2, strides=2)(z)
    z = Flatten()(z)

    out = Concatenate()([x, y, z])
    out = Dense(120, activation='relu')(out)
    out = Dropout(0.1)(out)
    out = Dense(84, activation='relu')(out)
    out = Dense(8, activation='softmax')(out)

    return Model(input, out, name="Model8")


def UnitResNet(x,filters, kernel_size=3, pool=False):

    res = x
    if pool:
        x = MaxPooling2D(pool_size=(2, 2))(x)
        res = Convolution2D(filters=filters,kernel_size=[1,1],strides=(2,2),padding="same")(res)
        
    out = BatchNormalization()(x)
    out = Activation("relu")(out)
    out = Convolution2D(filters=filters, kernel_size=kernel_size, strides=[1, 1], padding="same")(out)

    out = BatchNormalization()(out)
    out = Activation("relu")(out)
    out = Convolution2D(filters=filters, kernel_size=kernel_size, strides=[1, 1], padding="same")(out)

    out = keras.layers.add([res,out])

    return out

def Model9ResidualNet(img_width, img_height):
    
    input = Input(shape=(img_width, img_height, 3)) #32x32x3

    net = Convolution2D(filters=16, kernel_size=[3, 3], strides=[1, 1], padding="same")(input)
    net = UnitResNet(net,16)
    net = UnitResNet(net,16)

    net = BatchNormalization()(net)
    net = Activation("relu")(net)
    net = Dropout(0.25)(net)

    net = AveragePooling2D(pool_size=(4,4))(net)
    net = Flatten()(net)
    out = Dense(8, activation='softmax')(net)

    return Model(input, out, name="Model9ResidualNet")
    
def Model9ResidualNetV2(img_width, img_height):
    
    input = Input(shape=(img_width, img_height, 3)) #32x32x3

    net = Convolution2D(filters=16, kernel_size=[3, 3], strides=[1, 1], padding="same")(input)
    net = UnitResNet(net,16)

    net = BatchNormalization()(net)
    net = Activation("relu")(net)
    net = Dropout(0.4)(net)

    net = AveragePooling2D(pool_size=(4,4))(net)
    net = Flatten()(net)
    out = Dense(8, activation='softmax')(net)

    return Model(input, out, name="Model9ResidualNetV2")

def Model9ResidualNetV2Dropout(img_width, img_height):
    
    input = Input(shape=(img_width, img_height, 3)) #32x32x3

    net = Convolution2D(filters=10, kernel_size=[3, 3], strides=[1, 1], padding="same")(input)
    net = UnitResNet(net,10)
    net = Dropout(0.5)(net)

    net = BatchNormalization()(net)
    net = Activation("relu")(net)
    net = Dropout(0.5)(net)

    net = AveragePooling2D(pool_size=(4,4))(net)
    net = Flatten()(net)
    net = Dropout(0.5)(net)    
    out = Dense(8, activation='softmax')(net)

    return Model(input, out, name="Model9ResidualNetV2Dropout")

def Model9ResidualNetV2DropoutV3(img_width, img_height):
    
    input = Input(shape=(img_width, img_height, 3)) #32x32x3
    
    net = Convolution2D(filters=8, kernel_size=[3, 3], strides=[1, 1], padding="same")(input)
    net = UnitResNet(net,8)
    net = Dropout(0.5)(net)

    net = Convolution2D(filters=10, kernel_size=[3, 3], strides=[1, 1], padding="same")(net)
    net = UnitResNet(net,10)
    net = Dropout(0.5)(net)

    net = BatchNormalization()(net)
    net = Activation("relu")(net)
    net = Dropout(0.5)(net)

    net = AveragePooling2D(pool_size=(4,4))(net)
    net = Flatten()(net)
    net = Dropout(0.5)(net)    
    out = Dense(8, activation='softmax')(net)

    return Model(input, out, name="Model9ResidualNetV2DropoutV3")

def Model9ResidualNetV2DropoutV4(img_width, img_height):
    
    input = Input(shape=(img_width, img_height, 3)) #32x32x3
    
    net = Convolution2D(filters=8, kernel_size=[3, 3], strides=[1, 1], padding="same")(input)
    net = UnitResNet(net,8)
    net = Dropout(0.5)(net)

    net = Convolution2D(filters=10, kernel_size=[3, 3], strides=[1, 1], padding="same")(net)
    net = UnitResNet(net,10)
    net = Dropout(0.5)(net)

    net = BatchNormalization()(net)
    net = Activation("relu")(net)
    net = Dropout(0.5)(net)

    net = AveragePooling2D(pool_size=(4,4))(net)
    net = Flatten()(net)
    net = Dropout(0.5)(net)    
    out = Dense(8, activation='softmax')(net)

    return Model(input, out, name="Model9ResidualNetV2DropoutV3")