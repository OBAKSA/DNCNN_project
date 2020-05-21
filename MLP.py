# Importing tensorflow
import tensorflow as tf
# importing the data
from tensorflow.examples.tutorials.mnist import input_data
# Importing some more libraries
import matplotlib.pyplot as plt
from numpy import loadtxt
import numpy as np
from pylab import rcParams
from sklearn import preprocessing
import cv2
import scipy
import skimage.measure
import os
from random import randint
import math

'''
mnist = input_data.read_data_sets("./mnist/data/", one_hot=True)
X_train = mnist.train.images
X_test = mnist.test.images
'''

'''
matdata = scipy.io.loadmat("./mnist/train_32x32")
temp = matdata['X']
X_train = []
X_train_temp = []

for i in range(temp.shape[3]):
    X_train_temp.append(temp[:,:,:,i])
    
for i in range(temp.shape[3]):
    img_gray = (skimage.measure.block_reduce(cv2.cvtColor(X_train_temp[i], cv2.COLOR_BGR2GRAY), (2,2), np.max)).reshape(256)
    X_train.append(img_gray)

X_train = np.asarray(X_train)
print(X_train.shape)
'''

patch_size = 17
mean = 0
stddev = 15

PSNR_0 = []
PSNR_1 = []

def psnr(img1, img2):
    mse = np.mean( (img1 - img2) ** 2 )
    if mse == 0:
        return 100
    PIXEL_MAX = 255.0
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))

def make_patch(img1):
    patch = patch_size 
    sero = img1.shape[0]
    garo = img1.shape[1]
    i = randint(0,sero-patch)
    j = randint(0,garo-patch)
    
    random = randint(0,1)
    if random == 1:        
        return img1[i:i+patch,j:j+patch]
    else:
        return np.fliplr(img1[i:i+patch,j:j+patch])
    
def choose_patch(img1,img2):
    patch = patch_size 
    sero = img1.shape[0]
    garo = img1.shape[1]
    i = randint(0,sero-patch)
    j = randint(0,garo-patch)
    
    return (img1[i:i+patch,j:j+patch], img2[i:i+patch,j:j+patch])
    
X_train = []
X_train_noisy = []

file_dir = './BSDS300-images/BSDS300/images/train'
file_names = os.listdir(file_dir)
for name in file_names:
    BGR_img = cv2.imread(file_dir + '/' + name)
    img_gray = cv2.cvtColor(BGR_img, cv2.COLOR_BGR2GRAY)
    if img_gray.shape[0] > img_gray.shape[1]:
        img_gray = np.transpose(img_gray)
    img_gray = img_gray.reshape(321,481)
    #img_gray_noisy = img_gray + noise
    
    X_train.append(img_gray)
    #X_train_noisy.append(img_gray_noisy)

X_train = np.asarray(X_train)
#X_train_noisy = np.asarray(X_train_noisy)
X_train = X_train
#X_train_noisy = X_train_noisy/255.

# Deciding how many nodes each layer should have
n_nodes_inpl = patch_size*patch_size
n_nodes_hl1  = 711
n_nodes_hl2  = 711
n_nodes_hl3  = 711
n_nodes_outl = patch_size*patch_size
# hidden layers
hidden_1_layer_vals = {'weights':tf.Variable(tf.random_normal([n_nodes_inpl,n_nodes_hl1])),'biases':tf.Variable(tf.zeros([n_nodes_hl1]))  }
hidden_2_layer_vals = {'weights':tf.Variable(tf.random_normal([n_nodes_hl1, n_nodes_hl2])),'biases':tf.Variable(tf.zeros([n_nodes_hl2]))  }
hidden_3_layer_vals = {'weights':tf.Variable(tf.random_normal([n_nodes_hl2, n_nodes_hl3])),'biases':tf.Variable(tf.zeros([n_nodes_hl3]))  }
output_layer_vals = {'weights':tf.Variable(tf.random_normal([n_nodes_hl3,n_nodes_outl])),'biases':tf.Variable(tf.zeros([n_nodes_outl])) }

# 32*32

input_layer = tf.placeholder('float', [None, patch_size*patch_size])
layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(input_layer,hidden_1_layer_vals['weights']),hidden_1_layer_vals['biases']))
layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1,hidden_2_layer_vals['weights']), hidden_2_layer_vals['biases']))
layer_3 = tf.nn.sigmoid(tf.add(tf.matmul(layer_2,hidden_3_layer_vals['weights']), hidden_3_layer_vals['biases']))
output_layer = (tf.add(tf.matmul(layer_3,output_layer_vals['weights']),output_layer_vals['biases']))

output_true = tf.placeholder('float', [None, patch_size*patch_size])
# Cost Function
meansq = tf.reduce_mean(tf.square(output_layer - output_true))
# Optimizer
learn_rate = 0.001
optimizer = tf.train.AdamOptimizer(learning_rate=learn_rate).minimize(meansq)
#optimizer = tf.train.GradientDescentOptimizer(learning_rate=learn_rate).minimize(meansq)

def test():
    BGR_img = cv2.imread('Lena.png')
    BGR_img1 = cv2.imread('Man.png')
    
    img_gray = cv2.cvtColor(BGR_img, cv2.COLOR_BGR2GRAY)
    img_gray1 = cv2.cvtColor(BGR_img1, cv2.COLOR_BGR2GRAY)
    mean = 0
    sigma_noise = 15
    
    gaussian =  np.random.normal(mean, sigma_noise, img_gray.shape) 
    
    noisy_img = np.zeros(img_gray.shape, np.float32)
    noisy_img[:,:] = img_gray[:,:] + gaussian
    noisy_img1 = np.zeros(img_gray1.shape, np.float32)
    noisy_img1[:,:] = img_gray1[:,:] + gaussian
    
    npad_gray = [(0,0),(0,0)]
    img_gray_padded = np.pad(noisy_img, npad_gray, mode='constant')    
    img_gray_padded = img_gray_padded/255.
    img_gray_padded1 = np.pad(noisy_img1, npad_gray, mode='constant')    
    img_gray_padded1 = img_gray_padded1/255.
    
    # STRIDE = 3
    
    avg_cnt = np.zeros((512,512))
    counter = np.ones((patch_size,patch_size))
    output_full = np.zeros((512,512))
    output_full1 = np.zeros((512,512))  
    
    for i in range(166):
        for j in range(166):
            output_denoised_patch = sess.run(output_layer, feed_dict={input_layer:[img_gray_padded[3*i:3*i+patch_size,3*j:3*j+patch_size].reshape(patch_size*patch_size)]}).reshape(patch_size,patch_size)
            output_denoised_patch1 = sess.run(output_layer, feed_dict={input_layer:[img_gray_padded1[3*i:3*i+patch_size,3*j:3*j+patch_size].reshape(patch_size*patch_size)]}).reshape(patch_size,patch_size)
            output_full[3*i:3*i+patch_size,3*j:3*j+patch_size] += output_denoised_patch
            output_full1[3*i:3*i+patch_size,3*j:3*j+patch_size] += output_denoised_patch1
            avg_cnt[3*i:3*i+patch_size,3*j:3*j+patch_size] += counter
    
    output_full_avg = (np.divide(output_full,avg_cnt))*255.
    output_full_avg1 = (np.divide(output_full1,avg_cnt))*255.
    
    rcParams['figure.figsize'] = 20,20
    
    plt.subplot(1, 3, 1)
    plt.imshow(img_gray, cmap=plt.cm.gray),plt.title('Original')
    plt.xticks([]), plt.yticks([])
    
    plt.subplot(1, 3, 2)
    plt.imshow(noisy_img, cmap=plt.cm.gray),plt.title('Noise Added')
    plt.xticks([]), plt.yticks([])
    
    plt.subplot(1, 3, 3)
    plt.imshow(output_full_avg, cmap=plt.cm.gray),plt.title('Filtered')
    plt.xticks([]), plt.yticks([])
    
    plt.show()
    
    plt.subplot(1, 3, 1)
    plt.imshow(img_gray1, cmap=plt.cm.gray),plt.title('Original')
    plt.xticks([]), plt.yticks([])
    
    plt.subplot(1, 3, 2)
    plt.imshow(noisy_img1, cmap=plt.cm.gray),plt.title('Noise Added')
    plt.xticks([]), plt.yticks([])
    
    plt.subplot(1, 3, 3)
    plt.imshow(output_full_avg1, cmap=plt.cm.gray),plt.title('Filtered')
    plt.xticks([]), plt.yticks([])
    
    plt.show()
    
    PSNR0 = psnr(img_gray,output_full_avg)
    PSNR1 = psnr(img_gray1,output_full_avg1)
    
    print(PSNR0)
    print(PSNR1)
    
    PSNR_0.append(PSNR0)
    PSNR_1.append(PSNR1)

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

batch_size = 300
hm_epochs = 250000
tot_images = X_train.shape[0]

for epoch in range(1,hm_epochs+1):
    epoch_loss = 0 
    for i in range(int(tot_images/batch_size)):
        lepoch_x = []
        lepoch_x_n = []
        epoch_x1 = X_train[ i*batch_size : (i+1)*batch_size ]
        
        for i in range(batch_size):
            patched_img = make_patch(epoch_x1[i])            
            noise = np.random.normal(mean, stddev, patched_img.shape)
            noised = patched_img + noise
    
            patched_img = patched_img/255
            noised = noised/255.
            
            lepoch_x.append(patched_img.reshape(patch_size*patch_size))
            lepoch_x_n.append(noised.reshape(patch_size*patch_size))    
        
        epoch_x = np.asarray(lepoch_x)
        epoch_x_n = np.asarray(lepoch_x_n)
        
        _, c = sess.run([optimizer, meansq],\
               feed_dict={input_layer: epoch_x_n, \
               output_true: epoch_x})
        epoch_loss += c
    
    if epoch%5000 == 0:
        print('EPOCH ' + str(epoch))
        test()
        '''
        patched_img = make_patch(X_train[8])
        noised = patched_img + np.random.normal(mean, stddev, patched_img.shape)
        
        max1 = np.max(np.max(patched_img))
        max2 = np.max(np.max(noised))        
        patched_img = patched_img/255.
        noised = noised/255.
        
        a_i = patched_img.reshape(patch_size*patch_size)
        any_image = noised.reshape(patch_size*patch_size)
        
        output_any_image = sess.run(output_layer,\
                           feed_dict={input_layer:[any_image]})
        
        rcParams['figure.figsize'] = 5,5
        
        # Noisy Image
        plt.subplot(1, 3, 1)
        plt.imshow(any_image.reshape(patch_size,patch_size),  cmap='Greys')
        plt.axis('off')
        # Ground Truth
        plt.subplot(1, 3, 2)
        plt.imshow(a_i.reshape(patch_size,patch_size),  cmap='Greys')
        plt.axis('off')
        # Denoised Image
        plt.subplot(1, 3, 3)
        plt.imshow(output_any_image.reshape(patch_size,patch_size),  cmap='Greys')
        plt.axis('off')
        plt.show()    
        '''
    if epoch%100 == 0 :
        print('Epoch', epoch, '/', hm_epochs, 'loss:',epoch_loss)


x = range(5000,5000*(len(PSNR_0)+1),5000)
y0 = PSNR_0
y1 = PSNR_1

rcParams['figure.figsize'] = 8,8
rcParams.update({'font.size': 12})

plt.plot(x,y0,'b',label='Lena',marker='^')
plt.plot(x,y1,'r',label='Man',marker='D')
plt.xlabel('Epochs')
plt.ylabel('PSNR(dB)')
plt.title('MLP(multilayer perceptron)')
plt.legend(loc='upper right')
plt.show()

