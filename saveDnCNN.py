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
'''
# define functions needed to construct layers of CNN
def conv(name, inputs, nums_out, ksize, strides, padding="SAME", is_SN=False):
    with tf.variable_scope(name):
        W = tf.get_variable("W", shape=[ksize, ksize, int(inputs.shape[-1]), nums_out], initializer=tf.truncated_normal_initializer(stddev=0.02))
        b = tf.get_variable("b", shape=[nums_out], initializer=tf.constant_initializer(0.))
        if is_SN:
            return tf.nn.conv2d(inputs, spectral_norm(name, W), [1, strides, strides, 1], padding) + b
        else:
            return tf.nn.conv2d(inputs, W, [1, strides, strides, 1], padding) + b
        
def spectral_norm(name, w, iteration=1):
    w_shape = w.shape.as_list()
    w = tf.reshape(w, [-1, w_shape[-1]])
    with tf.variable_scope(name, reuse=False):
        u = tf.get_variable("u", [1, w_shape[-1]], initializer=tf.truncated_normal_initializer(), trainable=False)
    u_hat = u
    v_hat = None

    def l2_norm(v, eps=1e-12):
        return v / (tf.reduce_sum(v ** 2) ** 0.5 + eps)

    for i in range(iteration):
        v_ = tf.matmul(u_hat, tf.transpose(w))
        v_hat = l2_norm(v_)
        u_ = tf.matmul(v_hat, w)
        u_hat = l2_norm(u_)
    sigma = tf.matmul(tf.matmul(v_hat, w), tf.transpose(u_hat))
    w_norm = w / sigma
    with tf.control_dependencies([u.assign(u_hat)]):
        w_norm = tf.reshape(w_norm, w_shape)
    return w_norm    
    
def batchnorm(x, train_phase, scope_bn):
    with tf.variable_scope(scope_bn, reuse=tf.AUTO_REUSE):
        beta = tf.get_variable(name='beta', shape=[x.shape[-1]], initializer=tf.constant_initializer([0.]), trainable=True)
        gamma = tf.get_variable(name='gamma', shape=[x.shape[-1]], initializer=tf.constant_initializer([1.]), trainable=True)
        batch_mean, batch_var = tf.nn.moments(x, [0, 1, 2], name='moments')
        ema = tf.train.ExponentialMovingAverage(decay=0.5)

        def mean_var_with_update():
            ema_apply_op = ema.apply([batch_mean, batch_var])
            with tf.control_dependencies([ema_apply_op]):
                return tf.identity(batch_mean), tf.identity(batch_var)

        mean, var = tf.cond(train_phase, mean_var_with_update,
                            lambda: (ema.average(batch_mean), ema.average(batch_var)))
        normed = tf.nn.batch_normalization(x, mean, var, beta, gamma, 1e-3)
    return normed
'''
'''
# construct net based on functions defined above
class net:
    def __init__(self, name):
        self.name = name

    def __call__(self, inputs, train_phase):
        with tf.variable_scope(self.name):
            inputs = tf.nn.relu(conv("conv0", inputs, 64, 3, 1))
            for d in np.arange(1, DEPTH):
                inputs = tf.nn.relu(batchnorm(conv("conv_" + str(d + 1), inputs, 64, 3, 1), train_phase, "bn" + str(d)))
            outputs = conv("conv" + str(DEPTH), inputs, IMG_C, 3, 1)
            return outputs
'''         
'''
with tf.variable_scope('block1'):
        output = tf.layers.conv2d(input, 64, 3, padding='same', activation=tf.nn.relu)
    for layers in range(2, 19+1):
        with tf.variable_scope('block%d' % layers):
            output = tf.layers.conv2d(output, 64, 3, padding='same', name='conv%d' % layers, use_bias=False)
            output = tf.nn.relu(tf.layers.batch_normalization(output, training=is_training))   
    with tf.variable_scope('block17'):
        output = tf.layers.conv2d(output, output_channels, 3, padding='same',use_bias=False)
    return input - output
'''
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

    def test(self, filename ="./TestingSet/08.png"):
        saver = tf.train.Saver()
        saver.restore(self.sess, "./save_para/DnCNN.ckpt")
        
        cleaned_img = np.reshape(np.resize(np.array(Image.open(filename)),(512,512)), [1, 512, 512, 1])
        noised_img = cleaned_img + np.random.normal(0, SIGMA_NOISE, cleaned_img.shape)
        [denoised_img] = self.sess.run([self.denoised_img], feed_dict={self.clean_img: cleaned_img, self.noised_img: noised_img, self.train_phase: False})
        compared = np.concatenate((cleaned_img[0, :, :, 0], noised_img[0, :, :, 0], denoised_img[0, :, :, 0]), 1)
        Image.fromarray(np.uint8(compared)).show()

        PSNR_noise = psnr(cleaned_img[0, :, :, 0],noised_img[0, :, :, 0])
        PSNR_denoised = psnr(cleaned_img[0, :, :, 0],denoised_img[0, :, :, 0])
        
        print(PSNR_noise)
        print(PSNR_denoised)
        

if __name__ == "__main__":
    dncnn = DnCNN()
    #dncnn.train()
    dncnn.test()

