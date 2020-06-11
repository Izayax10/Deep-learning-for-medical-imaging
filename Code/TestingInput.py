import h5py as hp 
import matplotlib.pyplot as plt
import numpy as np

filepathtest = 'C://Users//anibo//OneDrive//Desktop//Final Year Project//Code//trainDataSet.mat'
filepathtrain = 'C://Users//anibo//OneDrive//Desktop//Final Year Project//Code//testDataSet.mat'

f = hp.File(filepathtest, 'r')
arrays = {}
for k, v in f.items():
    arrays[k] = np.array(v)
x_train = np.array(arrays['imagesTrue'])
x_train_noisy = np.array(arrays['imagesRecon'])

x_train = x_train.astype('float32') / 255.
x_train_noisy = x_train_noisy.astype('float32') / 255.

plt.figure(figsize=(20, 4))
plt.imshow(x_train_noisy[1][:][:])
plt.show()
