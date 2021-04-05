#  IMPORTS
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import tensorflow as tf

from tensorflow.keras.layers import Input

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D
from keras.optimizers import SGD
from keras import backend as K

from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, TerminateOnNaN, TensorBoard, ReduceLROnPlateau

import fileManager as fm

# from sklearn.metrics import confusion_matrix, accuracy_score, f1_score

##### HYPER Parametros
training_size,test_size,validate_size = 0.6,0.2,0.2
INIT_LR = 0.0007
EPOCHS = 100
BS = 16

def cnn_model():
    model = Sequential()

    model.add(Conv2D(32, (3, 3), padding='same',
                     input_shape=(3, 128, 128),
                     activation='relu'))
    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))

    model.add(Conv2D(64, (3, 3), padding='same',
                     activation='relu'))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))

    model.add(Conv2D(128, (3, 3), padding='same',
                     activation='relu'))
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))

    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(43, activation='softmax'))
    return model

def main():
    K.set_image_data_format('channels_first')

    # fm.createCsvFile()
    # fm.dataAugmentation()
    # fm.organizeDataSet()

    print('Training things...')

    # create a data generator
    datagen = ImageDataGenerator()

    # load and iterate training dataset
    train_it = datagen.flow_from_directory('dataset/train/', class_mode='binary', batch_size=BS)
    # load and iterate validation dataset
    val_it = datagen.flow_from_directory('dataset/validation/', class_mode='binary', batch_size=BS)
    # load and iterate test dataset
    test_it = datagen.flow_from_directory('dataset/test/', class_mode='binary', batch_size=BS)
    
    """
    model = tf.keras.applications.ResNet50(
        include_top=False, # Se sim o input terá de ser (224, 224, 3)
        weights=None,
        input_tensor=Input(shape=(3, 128, 128)),
        input_shape=(3, 128, 128),
        pooling=None,
        classes=43
    #    **kwargs
    )"""

    model = cnn_model()

    model_checkpoint = ModelCheckpoint(
        filepath='VGG_epoch-{epoch:02d}_loss-{val_loss:.4f}.h5',
        monitor='val_loss',
        verbose=1,
        save_best_only=True,
        save_weights_only=False,
        mode='auto',
        period=20
    )

    terminate_on_nan = TerminateOnNaN()

    t_board = TensorBoard(
        log_dir='./logs',
        histogram_freq=0,
        batch_size=BS,
        write_graph=True,
        write_images=False,
        embeddings_freq=0,
        update_freq='epoch'
    )

    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.2,
        patience=10,
        min_lr=0.0001
    )

    callbacks = [model_checkpoint, t_board, terminate_on_nan, reduce_lr]

    # Optimizer

    model.compile(loss='categorical_crossentropy', metrics=['accuracy'])

    print("Training...")

    # fit model
    H = model.fit(train_it, steps_per_epoch=16, validation_data=val_it, validation_steps=8)
  
    # evaluate model
    loss = model.evaluate_generator(test_it, steps=24)

def train_validate_test_split(df, train_percent=.6, validate_percent=.2, seed=None):
    np.random.seed(seed)
    perm = np.random.permutation(df.index)
    m = len(df.index)
    train_end = int(train_percent * m)
    validate_end = int(validate_percent * m) + train_end
    train = df.iloc[perm[:train_end]]
    validate = df.iloc[perm[train_end:validate_end]]
    test = df.iloc[perm[validate_end:]]
    return train, validate, test


main()

"""

    ##### READ FILE
    df = pd.read_csv('dataset.csv', names=['path', 'class'])

    X_train, Y_train, X_test, Y_test = [], [], [], []
    for classId in range(df['class'].nunique()):
        classData = []
        classLabels = []
        for i, row in df.iterrows():
            if row['class'] == classId:
                classData.append(row['path'])
                classLabels.append(classId)
        (trainX, testX, trainY, testY) = train_test_split(classData, classLabels, test_size=test_size)
        X_train.append(trainX)
        X_test.append(testX)
        Y_train.append(trainY)
        Y_test.append(testY)

train, validate, test = \
              np.split(df.sample(frac=1, random_state=42), 
                       [int(.6*len(df)), int(.8*len(df))])

file = open("OCRDataSet\\OCRDataSet.txt", "r")




data = []
for str in file.read().split('\t'):
    try:
        data.append(int(str))
    except:
        print("Impossivel de ler inteiro: " + str.__repr__())


##### DATA SPLIT
# Matrix em que cada linha será uma letra
letters = [data[i : i + (50*50)] for i in range(0, len(data), (50*50))]

# Dividar dados (training_size)
count_t_d = int(20 * training_size)

# Shuffle data
tmp = []
for i in range(0, len(letters), 20):
    lst = letters[i : i + 20]
    random.shuffle(lst)
    tmp.extend(lst)
letters = tmp

# Matriz: cada linha é uma letra (36 classes)
training_data = [letters[i : i + count_t_d] for i in range(0, len(letters), 20)]
test_data = [letters[i : i + (20 - count_t_d)] for i in range(count_t_d, len(letters), 20)]

##### Carateristicas (FEATURES)
# -> numero de pixeis a preto
# -> Gaborr bank 

# AKA features
x_train, y_train = [], [] # x_train: nº de imagens de treino linhas, k (nº de carateristicas) carateristicas colunas
# y_train: labels corretos para cada linha
y_labels = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9'] # AKA y_train

# Obter features e construir os dados de treino
for i in range(36):
    feat, labels = fe.getFeaturesLst(training_data[i], y_labels[i])
    x_train.extend(feat)
    y_train.extend(labels)

##### TEST DATA

x_test, y_test = [], []

for i in range(36):
    feat, labels = fe.getFeaturesLst(test_data[i], y_labels[i])
    x_test.extend(feat)
    y_test.extend(labels)

###### TREINAR E CLASSIFICAR
##### SVM

clf = svm.SVC()
clf.fit(x_train, y_train)

y_pred = clf.predict(x_test)

##### DECISION TREE

clf1 = tree.DecisionTreeClassifier()
clf1.fit(x_train, y_train)

y_pred1 = clf1.predict(x_test)

##### KNN

neigh = KNeighborsClassifier(n_neighbors=3)
neigh.fit(x_train, y_train)

y_pred2 = neigh.predict(x_test)


##### ESTATISTICA

def printStatistic(name_algorithm, y_pred, y_test, showHeatMap=False):
    cf_matrix = confusion_matrix(y_test, y_pred.tolist())
    print("Accuracy (%s): %f" % (name_algorithm, accuracy_score(np.array(y_test), y_pred)))
    print("F1-score (%s) por classe:" % name_algorithm)
    print(f1_score(y_test, y_pred.tolist(), average=None))
    print("F1-score (%s) (média pesada): %f" % (name_algorithm, f1_score(y_test, y_pred.tolist(), average='weighted')))
    print(" ")
    if (showHeatMap):
        sns.heatmap(cf_matrix, annot=True)
        plt.show()

printStatistic("SVM", y_pred, y_test, True)
printStatistic("Decision Tree", y_pred1, y_test, True)
printStatistic("KNN", y_pred2, y_test, True)

##### VER FILTROS/LETRAS

# Dividir lista da letra pelas linhas (50x50)
letter_number = 20 * 18 + 10
letter = [letters[letter_number][i : i + 50] for i in range(0, len(letters[letter_number]), 50)]

# imgplot = plt.imshow(letter, cmap='gray')
# plt.show()




"""