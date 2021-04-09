#  IMPORTS
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

import tensorflow as tf

from tensorflow.keras.layers import Input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, TerminateOnNaN, TensorBoard, ReduceLROnPlateau

import fileManager as fm

##### HYPER Parametros
training_size,test_size,validate_size = 0.6,0.2,0.2
NUM_CLASSES = 43
INIT_LR = 0.0007
NUM_EPOCHS = 100
BS = 16
NUM_IMAGES = 1500*43

def main():

    # fm.createCsvFile()
    # fm.dataAugmentation()
    # fm.organizeDataSet()

    print('Training things...')

    # create a data generator
    datagen = ImageDataGenerator(rescale=1./255)

    train_it = datagen.flow_from_directory('dataset/train', batch_size=BS, target_size=(128, 128))
    val_it = datagen.flow_from_directory('dataset/validation', batch_size=BS, target_size=(128, 128))
    test_it = datagen.flow_from_directory('dataset/test', batch_size=BS, target_size=(128, 128))

    model = tf.keras.applications.VGG19(
        weights=None,
        input_tensor=Input(shape=(128, 128, 3)),
        input_shape=(128, 128, 3),
        classes=43
    #    **kwargs
    )

    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy'])

    model.summary()

    print("Training...")

    H = model.fit_generator(train_it, epochs=NUM_EPOCHS, validation_data=val_it)
  
    # evaluate model
    loss = model.evaluate_generator(test_it, steps=16)

    # plot the training loss and accuracy
    N = NUM_EPOCHS
    plt.style.use("ggplot")
    plt.figure()
    plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
    plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
    plt.plot(np.arange(0, N), H.history["accuracy"], label="train_acc")
    plt.plot(np.arange(0, N), H.history["val_accuracy"], label="val_acc")
    plt.title("Training Loss and Accuracy on Dataset")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss/Accuracy")
    plt.legend(loc="lower left")
    plt.savefig("plot.png")

   
main()
