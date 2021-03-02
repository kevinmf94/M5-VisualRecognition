from keras.layers import Dense, Flatten
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adagrad, RMSprop, SGD
from keras.callbacks.tensorboard_v2 import TensorBoard
from keras.callbacks.callbacks import EarlyStopping, LambdaCallback
from keras.engine.sequential import Sequential
from keras.layers.pooling import AveragePooling2D, MaxPooling2D
from keras.layers.core import Dropout
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Convolution2D
from matplotlib import pyplot as plt
from keras.engine.training import Model
from keras.engine.input_layer import Input
from keras.layers.merge import Average, Concatenate
from models import *
from keras.utils.vis_utils import plot_model

#DATASET_TRAIN = "/Users/kevinmartinfernandez/Workspace/Master/M3/CNN/MIT_split/train"
#DATASET_TEST = "/Users/kevinmartinfernandez/Workspace/Master/M3/CNN/MIT_split/test"
DATASET_TRAIN = '/home/mcv/datasets/MIT_split/train'
DATASET_TEST = '/home/mcv/datasets/MIT_split/test'
EPOCHS = 300
batch_size = 200
img_width = 128
img_height = 128
validation_samples=807

print("Start script!", flush=True)

#Data generator
datagen = ImageDataGenerator(
    shear_range=1.,
    channel_shift_range=1.,
    horizontal_flip=True,
    rescale=1./255)

train_generator = datagen.flow_from_directory(DATASET_TRAIN,
        shuffle=True,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode='categorical')

validation_generator = datagen.flow_from_directory(DATASET_TEST,
        shuffle=True,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode='categorical')

print("Create model!", flush=True)
# Create model

print("{}x{}".format(img_width, img_height))
model = Model9ResidualNetV2DropoutV3(img_width, img_height)

model.compile(optimizer=SGD(learning_rate=0.1), loss='categorical_crossentropy', metrics=['accuracy'])

#Summary
model.summary()

plot_model(model, to_file='Model9ResidualNetV2DropoutV3.png', show_shapes=True, show_layer_names=True)
exit()

#Callbacks
#tbCallBack = TensorBoard(log_dir='/home/group08/work/logs', histogram_freq=0, write_graph=True)
logCallback = LambdaCallback(on_epoch_end=lambda epoch, logs: print(epoch, logs, flush=True))
earlyCallback = EarlyStopping(patience=2)

print("Start training", flush=True)

history=model.fit_generator(train_generator,
        #steps_per_epoch=(int(train_generator.n//batch_size)+1),
        steps_per_epoch = batch_size,
        epochs=EPOCHS,
        validation_data=validation_generator,
        verbose=1,
        validation_steps= batch_size, 
        callbacks=[logCallback, earlyCallback])

print("Finish training", flush=True)

if True:
  # summarize history for accuracy
  plt.plot(history.history['accuracy'])
  plt.plot(history.history['val_accuracy'])
  plt.title('model accuracy')
  plt.ylabel('accuracy')
  plt.ylim(bottom=0.0, top=1.0)
  plt.xlabel('epoch')
  plt.legend(['train', 'validation'], loc='upper left')
  plt.savefig('accuracy.jpg')
  plt.close()
  # summarize history for loss
  plt.plot(history.history['loss'])
  plt.plot(history.history['val_loss'])
  plt.title('model loss')
  plt.ylabel('loss')
  plt.ylim(bottom=0)
  plt.xlabel('epoch')
  plt.legend(['train', 'validation'], loc='upper left')
  plt.savefig('loss.jpg')

  print("Generated plots", flush=True)
