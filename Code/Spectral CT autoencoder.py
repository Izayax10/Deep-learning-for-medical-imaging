from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D
from keras.models import Model
from keras import backend as K
from keras.datasets import mnist
from keras.datasets import fashion_mnist
import numpy as np
import h5py as hp
import matplotlib.pyplot as plt
import tensorflow as tf
import cv2
xdim = 64
ydim = 64

input_img = Input(shape=(xdim, ydim, 3))  
x = Conv2D(256, (3, 3), activation='relu', padding='same')(input_img)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
encoded = MaxPooling2D((2, 2), padding='same')(x)

# at this point the representation is (7, 7, 32)

x = Conv2D(32, (3, 3), activation='relu', padding='same')(encoded)
x = UpSampling2D((2, 2))(x)
x = Conv2D(256, (3, 3), activation='relu', padding='same')(x)
x = UpSampling2D((2, 2))(x)
decoded = Conv2D(3, (3, 3), activation='sigmoid', padding='same')(x)
print(x)
# autoencoder = Model(input_img, decoded)
# autoencoder.compile(optimizer='adam', loss='mse')


# filepath = "C://Users//anibo//AppData//Local//atom//app-1.45.0//trainDataSet.mat"
# arrays = {}
# f = hp.File(filepath, 'r')
# for k, v in f.items():
#     arrays[k] = np.array(v)
# x_train = np.array(arrays['imagesTrue'])
# filepath = "C://Users//anibo//AppData//Local//atom//app-1.45.0//testDataSet.mat"
# arrays = {}
# f = hp.File(filepath, 'r')
# for k, v in f.items():
#     arrays[k] = np.array(v)
# x_test = np.array(arrays['imagesTrue'])

filepathtrain = 'C://Users//anibo//OneDrive//Desktop//Final Year Project//Code//trainDataSet.mat'
filepathtest = 'C://Users//anibo//OneDrive//Desktop//Final Year Project//Code//testDataSet.mat'
filepathtrainhead = 'C://Users//anibo//OneDrive//Desktop//Final Year Project//Code//trainDataSethead.mat'

f = hp.File(filepathtrain, 'r')
arraystrain = {}
for k, v in f.items():
    arraystrain[k] = np.array(v)
x_train = np.array(arraystrain['imagesTrue'])
x_train_noisy = np.array(arraystrain['imagesRecon'])

f = hp.File(filepathtest, 'r')
arraystest = {}
for k, v in f.items():
    arraystest[k] = np.array(v)
x_test = np.array(arraystest['imagesTrue'])
x_test_noisy = np.array(arraystest['imagesRecon'])

f = hp.File(filepathtrainhead, 'r')
arraystesthead = {}
for k, v in f.items():
    arraystesthead[k] = np.array(v)
x_test_head = np.array(arraystesthead['imagesTrue'])
x_test_head_noisy = np.array(arraystesthead['imagesRecon'])

x_train = x_train.astype('float32') #/ 255.
x_train_noisy = x_train_noisy.astype('float32') #/ 255.
x_train = np.reshape(x_train, (6000, xdim, ydim, 1))  
x_train_noisy = np.reshape(x_train_noisy, (6000, xdim, ydim, 1)) 

x_test = x_test.astype('float32') #/ 255.
x_test_noisy = x_test_noisy.astype('float32') #/ 255.
x_test = np.reshape(x_test, (1200, xdim, ydim, 1))  
x_test_noisy = np.reshape(x_test_noisy, (1200, xdim, ydim, 1))  

x_valid = x_test[1000:1200,:,:,:]
x_test = x_test[1:1000,:,:,:]
x_valid_noisy = x_test_noisy[1000:1200,:,:,:]
x_test_noisy = x_test_noisy[1:1000,:,:,:]

# x_test_head = x_test_head.astype('float32') 
# x_test_head_noisy = x_test_head_noisy.astype('float32')
# x_test_head = np.reshape(x_test_head, (len(x_test_head), 256, 256, 1)) 
# x_test_head_noisy = np.reshape(x_test_head_noisy, (len(x_test_head_noisy), 256, 256, 1)) 

# x_test = np.array(tf.image.grayscale_to_rgb(tf.convert_to_tensor(x_test)))
# x_test_noisy = np.array(tf.image.grayscale_to_rgb(tf.convert_to_tensor(x_test_noisy)))
# x_train = np.array(tf.image.grayscale_to_rgb(tf.convert_to_tensor(x_train)))
# x_train_noisy = np.array(tf.image.grayscale_to_rgb(tf.convert_to_tensor(x_train_noisy)))
# x_test_head = np.array(tf.image.grayscale_to_rgb(tf.convert_to_tensor(x_test_head)))
# x_test_head_noisy = np.array(tf.image.grayscale_to_rgb(tf.convert_to_tensor(x_test_head_noisy)))

x_test_rgb = np.empty(shape=(len(x_test),64,64,3))
x_test_noisy_rgb = np.empty(shape=(len(x_test_noisy),64,64,3))
x_valid_noisy_rgb = np.empty(shape=(len(x_valid),64,64,3))
x_valid_rgb = np.empty(shape=(len(x_valid_noisy),64,64,3))
x_train_rgb = np.empty(shape=(len(x_train),64,64,3))
x_train_noisy_rgb = np.empty(shape=(len(x_train_noisy),64,64,3))

for i in range(200):
    x_valid_rgb[i] = cv2.cvtColor(x_valid[i,:,:],cv2.COLOR_GRAY2RGB)
    x_valid_noisy_rgb[i] = cv2.cvtColor(x_valid_noisy[i,:,:],cv2.COLOR_GRAY2RGB)
for i in range(999):
    x_test_rgb[i] = cv2.cvtColor(x_test[i,:,:],cv2.COLOR_GRAY2RGB)
    x_test_noisy_rgb[i] = cv2.cvtColor(x_test_noisy[i,:,:],cv2.COLOR_GRAY2RGB)
for i in range(6000):
    x_train_rgb[i] = cv2.cvtColor(x_train[i,:,:],cv2.COLOR_GRAY2RGB)
    x_train_noisy_rgb[i] = cv2.cvtColor(x_train_noisy[i,:,:],cv2.COLOR_GRAY2RGB)

x_test = x_test_rgb
x_test_noisy = x_test_noisy_rgb
x_train = x_train_rgb
x_train_noisy = x_train_noisy_rgb
x_valid = x_valid_rgb
x_valid_noisy = x_valid_noisy_rgb

autoencoder = Model(input_img, decoded)
autoencoder.compile(optimizer='adam', loss='binary_crossentropy')


autoencoder.fit(x_train_noisy_rgb, x_train_rgb,
                epochs=5,
                batch_size=5,
                shuffle=True,
                validation_data=(x_valid_noisy_rgb, x_valid_rgb))

decoded_imgs = autoencoder.predict(x_test_noisy_rgb)
x_test_rgb = x_test_rgb.astype('float32')
decoded_imgs = decoded_imgs.astype('float32')
y = tf.convert_to_tensor(x_test_rgb)
z = tf.convert_to_tensor(decoded_imgs)

ss = tf.image.ssim(y,z,max_val=255)
ps1 = tf.image.psnr(y, z, max_val=255)
ms = (np.square(y - z)).mean(axis=None)
ss = np.sum(ss)/1000
ps1 = np.sum(ps1)/1000
print(ss)
print(ps1)
print(ms)
n = 10
plt.figure(figsize=(20, 4))
for i in range(n):
    # display original
    ax = plt.subplot(2, n,n+ i+1)

    plt.imshow(decoded_imgs[i].reshape(64,64,3))
    #plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # display reconstruction
    ax = plt.subplot(2, n, i + 1)
    plt.imshow(x_test_noisy_rgb[i].reshape(64, 64,3))
    #plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()
