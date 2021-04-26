#  IMPORTS
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

import tensorflow as tf

from tensorflow.keras.layers import Input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, TerminateOnNaN, TensorBoard, ReduceLROnPlateau
from tensorflow.keras.optimizers import SGD, RMSprop

import fileManager as fm

##### HYPER Parameters
training_size,test_size,validate_size = 0.6,0.2,0.2
NUM_CLASSES = 43
INIT_LR = 0.01
NUM_EPOCHS = 100
BS = 32
NUM_IMAGES = 1500*43

def main():

    fm.createCsvFile()
    fm.dataAugmentation()
    fm.organizeDataSet()

    print('Training things...')

    # create a data generator
    datagen = ImageDataGenerator(rescale=1./255)

    train_it = datagen.flow_from_directory('dataset/train', batch_size=BS, target_size=(128, 128))
    val_it = datagen.flow_from_directory('dataset/validation', batch_size=BS, target_size=(128, 128))
    test_it = datagen.flow_from_directory('dataset/test', batch_size=BS, target_size=(128, 128))

    model = tf.keras.applications.VGG19(
        weights=None,
        input_tensor=Input(shape=(128, 128, 3)),
        pooling="avg",
        classifier_activation="softmax",
        classes=43
    )

    rms = RMSprop(learning_rate=1e-3)
    # rms = RMSprop(learning_rate=1e-5)

    es = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3)

    callbacks = [es]

    model.compile(
        optimizer=rms,
        loss='categorical_crossentropy',
        metrics=['accuracy'])

    model.summary()

    print("Training...")

    H = model.fit(train_it, 
        epochs=NUM_EPOCHS, 
        validation_data=val_it,
        callbacks=callbacks)
  
    # evaluate model
    test_loss = model.evaluate(test_it)
    test_acc = model.predict(test_it, steps =np.ceil(len(test_it.filenames)/BS))

    model.save('model')

    print('*-----*')
    print('Test loss: ', test_loss)
    print('Test accuracy: ', test_acc)
    print('*-----*')

    # plot the training loss and accuracy
    N = NUM_EPOCHS
    plt.style.use("ggplot")
    plt.figure()
    plt.plot(H.history["loss"], label="train_loss")
    plt.plot(H.history["val_loss"], label="val_loss")
    plt.plot(H.history["accuracy"], label="train_acc")
    plt.plot(H.history["val_accuracy"], label="val_acc")
    plt.title("Training Loss and Accuracy on Dataset")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss/Accuracy")
    plt.legend(loc="lower left")
    plt.savefig("plot.png")

   
main()
