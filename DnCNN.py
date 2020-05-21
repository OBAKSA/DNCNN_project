# -*- coding: utf-8 -*-
"""
Created on Wed Oct 23 19:50:59 2019

@author: yjymo
"""
from PIL import Image
import os
import numpy as np
import tensorflow as tf
import math
from matplotlib import pyplot as plt

list_PSNR_noise = []
list_PSNR_denoised = []

IMG_H = 40
IMG_W = 40
IMG_C = 1
DEPTH = 16
BATCH_SIZE = 64
EPOCHS = 20
SIGMA_NOISE = 25
EPSILON = 1e-10

# calculates PSNR
def psnr(img1, img2):
    mse = np.mean( (img1 - img2) ** 2 )
    if mse == 0:
        return 100
    PIXEL_MAX = 255.0
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))

# define net using Relu, BN, conv
class net:
    def __init__(self, name):
        self.name = name

    def __call__(self, inputs, train_phase):
        with tf.variable_scope(self.name):
            inputs = tf.layers.conv2d(inputs, 64, 3, padding='same', activation = tf.nn.relu)
            for d in np.arange(1, DEPTH):
                inputs = tf.layers.conv2d(inputs, 64, 3, padding='same', name='conv_' + str(d+1), use_bias=False)
                inputs = tf.nn.relu(tf.layers.batch_normalization(inputs, training=True))
            outputs = tf.layers.conv2d(inputs, IMG_C, 3, padding='same', use_bias=False)
            return outputs
        
# class to train/test the net of DnCNN
class DnCNN:
    def __init__(self):
        self.clean_img = tf.placeholder(tf.float32, [None, None, None, IMG_C])
        self.noised_img = tf.placeholder(tf.float32, [None, None, None, IMG_C])
        self.train_phase = tf.placeholder(tf.bool)
        dncnn = net("DnCNN")
        self.res = dncnn(self.noised_img, self.train_phase)
        self.denoised_img = self.noised_img - self.res
        self.loss = tf.reduce_mean(tf.reduce_sum(tf.square(self.res - (self.noised_img - self.clean_img)), [1, 2, 3]))
        self.Opt = tf.train.AdamOptimizer(1e-3).minimize(self.loss)
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

    def train(self):
        filepath = "./TrainingSet/"
        filenames = os.listdir(filepath)
        saver = tf.train.Saver()
        for epoch in range(EPOCHS):
            for i in range(int(len(filenames)/BATCH_SIZE)):
                cleaned_batch = np.zeros([BATCH_SIZE, IMG_H, IMG_W, IMG_C])
                for idx, filename in enumerate(filenames[i*BATCH_SIZE:i*BATCH_SIZE+BATCH_SIZE]):
                    cleaned_batch[idx, :, :, 0] = np.array(Image.open(filepath+filename))
                noised_batch = cleaned_batch + np.random.normal(0, SIGMA_NOISE, cleaned_batch.shape)
                self.sess.run(self.Opt, feed_dict={self.clean_img: cleaned_batch, self.noised_img: noised_batch, self.train_phase: True})
                if i % 10 == 0:
                    [loss, denoised_img] = self.sess.run([self.loss, self.denoised_img], feed_dict={self.clean_img: cleaned_batch, self.noised_img: noised_batch, self.train_phase: False})
                    print("Epoch: %d, Step: %d, Loss: %g"%(epoch, i, loss))
                    compared = np.concatenate((cleaned_batch[0, :, :, 0], noised_batch[0, :, :, 0], denoised_img[0, :, :, 0]), 1)
                    Image.fromarray(np.uint8(compared)).save("./TrainingResults/"+str(epoch)+"_"+str(i)+".jpg")
                if i % 420 == 0:
                    saver.save(self.sess, "./save_para/DnCNN.ckpt")
            np.random.shuffle(filenames)
            
            #DnCNN.test()
            
            saver.restore(self.sess, "./save_para/DnCNN.ckpt")
            cleaned_img = np.reshape(np.resize(np.array(Image.open("./TestingSet/08.png")),(512,512)), [1, 512, 512, 1])
            noised_img = cleaned_img + np.random.normal(0, SIGMA_NOISE, cleaned_img.shape)
            [denoised_img] = self.sess.run([self.denoised_img], feed_dict={self.clean_img: cleaned_img, self.noised_img: noised_img, self.train_phase: False})
            compared = np.concatenate((cleaned_img[0, :, :, 0], noised_img[0, :, :, 0], denoised_img[0, :, :, 0]), 1)
            Image.fromarray(np.uint8(compared)).show()
            
            PSNR_noise = psnr(cleaned_img[0, :, :, 0],noised_img[0, :, :, 0])
            PSNR_denoised = psnr(cleaned_img[0, :, :, 0],denoised_img[0, :, :, 0])
            print('EPOCH : ' + str(epoch))
            print(PSNR_noise)
            print(PSNR_denoised)
            list_PSNR_noise.append(PSNR_noise)
            list_PSNR_denoised.append(PSNR_denoised)
            
        x = range(1,21)
        y = list_PSNR_denoised
        
        plt.plot(x,y,'r',label='with RL, with BN', marker='^')
        plt.xlabel('Epochs')
        plt.ylabel('PSNR(dB)')
        plt.title('DnCNN with Adam Optimizer')
        plt.legend(loc='lower right')
        plt.show()

    def test(self, filename ="./TestingSet/eiffel.jpg"):
        saver = tf.train.Saver()
        saver.restore(self.sess, "./save_para/DnCNN.ckpt")
        
        file = np.array(Image.open(filename))
        filesize = (file.shape)
        fileR = file[:,:,0]
        fileG = file[:,:,1]
        fileB = file[:,:,2]
                
        cleaned_img = np.reshape(np.resize(fileR,(filesize[0],filesize[1])), [1, filesize[0], filesize[1], 1])
        noised_img = cleaned_img + np.random.normal(0, SIGMA_NOISE-25, cleaned_img.shape)
        [denoised_img] = self.sess.run([self.denoised_img], feed_dict={self.clean_img: cleaned_img, self.noised_img: noised_img, self.train_phase: False})
        #compared = np.concatenate((cleaned_img[0, :, :, 0], noised_img[0, :, :, 0], denoised_img[0, :, :, 0]), 1)
        #Image.fromarray(np.uint8(compared)).show()

        #PSNR_noise = psnr(cleaned_img[0, :, :, 0],noised_img[0, :, :, 0])
        #PSNR_denoised = psnr(cleaned_img[0, :, :, 0],denoised_img[0, :, :, 0])
        
        #print(PSNR_noise)
        #print(PSNR_denoised)    
        R = denoised_img[0,:,:,0]
    
        cleaned_img = np.reshape(np.resize(fileG,(filesize[0],filesize[1])), [1, filesize[0], filesize[1], 1])
        noised_img = cleaned_img + np.random.normal(0, SIGMA_NOISE-25, cleaned_img.shape)
        [denoised_img] = self.sess.run([self.denoised_img], feed_dict={self.clean_img: cleaned_img, self.noised_img: noised_img, self.train_phase: False})
        #compared = np.concatenate((cleaned_img[0, :, :, 0], noised_img[0, :, :, 0], denoised_img[0, :, :, 0]), 1)
        #Image.fromarray(np.uint8(compared)).show()

        #PSNR_noise = psnr(cleaned_img[0, :, :, 0],noised_img[0, :, :, 0])
        #PSNR_denoised = psnr(cleaned_img[0, :, :, 0],denoised_img[0, :, :, 0])
        
        #print(PSNR_noise)
        #print(PSNR_denoised)
        G = denoised_img[0,:,:,0]
        
        cleaned_img = np.reshape(np.resize(fileB,(filesize[0],filesize[1])), [1, filesize[0], filesize[1], 1])
        noised_img = cleaned_img + np.random.normal(0, SIGMA_NOISE-25, cleaned_img.shape)
        [denoised_img] = self.sess.run([self.denoised_img], feed_dict={self.clean_img: cleaned_img, self.noised_img: noised_img, self.train_phase: False})
        #compared = np.concatenate((cleaned_img[0, :, :, 0], noised_img[0, :, :, 0], denoised_img[0, :, :, 0]), 1)
        #Image.fromarray(np.uint8(compared)).show()

        #PSNR_noise = psnr(cleaned_img[0, :, :, 0],noised_img[0, :, :, 0])
        #PSNR_denoised = psnr(cleaned_img[0, :, :, 0],denoised_img[0, :, :, 0])
        
        #print(PSNR_noise)
        #print(PSNR_denoised)
        B = denoised_img[0,:,:,0]
        
        RGB = np.zeros([filesize[0],filesize[1],3])
        RGB[:,:,0] = np.clip(R,0,255)
        RGB[:,:,1] = np.clip(G,0,255)
        RGB[:,:,2] = np.clip(B,0,255)
        
        img_file = Image.fromarray(np.uint8(file))
        img_denoised = Image.fromarray(np.uint8(RGB))
        
        img_file.show()
        img_denoised.show()
        
        img_file.save('eiffel_original.png')
        img_denoised.save('eiffel_denoised.png')
        
if __name__ == "__main__":
    dncnn = DnCNN()
    #dncnn.train()
    dncnn.test()

