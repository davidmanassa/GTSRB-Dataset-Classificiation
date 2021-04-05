"""

Este código serve para obtermos um ficheiro CSV no formanto: nome_ficheiro (path), classe

"""

import os
from os import listdir
from os.path import isfile, join
import shutil
import csv
import Augmentor
import pandas as pd
import numpy as np

def createCsvFile(csv_file_name='dataset.csv', dataset_folder='GTSRB'):

    print('Creating csv file...')

    onlyfiles = [f for f in listdir(dataset_folder) if isfile(join(dataset_folder, f))]

    file = open(csv_file_name, 'w', newline='')
    writer = csv.writer(file, delimiter=",")

    for f in onlyfiles:
        # numeroDeClasse_Numero

        splitted = f.split('.')[0].split('_')
        writer.writerow(('GTSRB/' + f, int(splitted[0])))

    return csv_file_name

def dataAugmentation(csvFile='dataset.csv', normal_folder='GTSRB', temp_folder='temp_folder'):
    """ 
        Aqui vou verificar e colocar o dataset com o mesmo numero de dados usando Data Augmentation
    """
    print('Data augmentation...')
    
    df = pd.read_csv('dataset.csv', names=['path', 'class'])
    
    countDataset = [0 for _ in range(df['class'].nunique())]
    for i, row in df.iterrows():
        countDataset[row['class']] += 1

    max = 0
    for i in countDataset:
        if i > max:
            max = i

    print(countDataset)

    if not os.path.exists(temp_folder):
        os.makedirs(temp_folder)
    for classId, classCount in enumerate(countDataset):
        if classCount < max:
            for i, row in df.iterrows():
                if row['class'] == classId:
                    shutil.move(row['path'], temp_folder)
            # Data augmentation
            p = Augmentor.Pipeline(temp_folder)
            p.rotate(probability=0.7, max_left_rotation=15, max_right_rotation=15)
            p.zoom(probability=0.5, min_factor=1.1, max_factor=1.5)
            p.skew(0.4, 0.5)
            p.sample(max - classCount)
            # Re organize
            new_files = os.listdir(temp_folder + '/output')
            for i, f in enumerate(new_files):
                n = i + classCount + 1
                new_name = str(classId) + '_' + str(n) + '.jpg'
                try:
                    os.rename(temp_folder + '/output/' + f, normal_folder + '/' + new_name)
                except Exception:
                    print('Fail rename? ' + new_name)
            files = os.listdir(temp_folder)
            for f in files:
                if (f != 'output'):
                    try:
                        shutil.move(temp_folder + '/' + f, normal_folder)
                    except Exception:
                        print('Fail? ' + f)


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

def organizeDataSet(csvFile='dataset.csv', train_size = 0.6, validade_size = 0.2):
    """
        Dividir dados de cada classe
        Atualizar path

        Não atualizamos o csv
    """
    print('Organizing dataset...')

    df = pd.read_csv(csvFile, names=['path', 'class'])

    numberOfClasses = df['class'].nunique()

    if not os.path.exists('dataset'):
        os.makedirs('dataset')
    else:
        return
    if not os.path.exists('dataset/test'):
        os.makedirs('dataset/test')
    if not os.path.exists('dataset/train'):
        os.makedirs('dataset/train')
    if not os.path.exists('dataset/validation'):
        os.makedirs('dataset/validation')

    for classId in range(numberOfClasses):

        if not os.path.exists('dataset/test/' + str(classId)):
            os.makedirs('dataset/test/' + str(classId))
        if not os.path.exists('dataset/train/' + str(classId)):
            os.makedirs('dataset/train/' + str(classId))
        if not os.path.exists('dataset/validation/' + str(classId)):
            os.makedirs('dataset/validation/' + str(classId))

        data = []

        for lineI, row in df.iterrows():

            if row['class'] == classId:

                data.append([row['path'], row['class']])

        temp_df = pd.DataFrame(data, columns = ['path', 'class'])

        train, validation, test = train_validate_test_split(temp_df, train_size, validade_size)

        for lineI, row in train.iterrows():

            file_name = row['path'].split('/')[len(row['path'].split('/')) - 1]
            shutil.copyfile(row['path'], 'dataset/train/' + str(classId) + '/' + file_name)
            
            # O CSV não será mais necessário, porém podiamos o atualizar aqui, se necessário.

        for lineI, row in test.iterrows():

            file_name = row['path'].split('/')[len(row['path'].split('/')) - 1]
            shutil.copyfile(row['path'], 'dataset/test/' + str(classId) + '/' + file_name)

        for lineI, row in validation.iterrows():

            file_name = row['path'].split('/')[len(row['path'].split('/')) - 1]
            shutil.copyfile(row['path'], 'dataset/validation/' + str(classId) + '/' + file_name)




