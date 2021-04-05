import os, shutil


lst = os.listdir('GTSRB')

for i in range(1500):
    if '25_' + str(i) + '.jpg' not in lst:
        print('25_' + str(i) + '.jpg')


