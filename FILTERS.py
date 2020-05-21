# -*- coding: utf-8 -*-
"""
Created on Mon Aug  5 20:35:32 2019

@author: yjymo
"""
import os
import math
import cv2
import numpy as np
from matplotlib import pyplot as plt
from multiprocessing import Pool

def psnr(img1, img2):
    mse = np.mean( (img1 - img2) ** 2 )
    if mse == 0:
        return 100
    PIXEL_MAX = 255.0
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))

def euclidean_distance(arr1,arr2):
    diff_arr = np.subtract(arr1,arr2)
    return math.sqrt(np.sum(np.multiply(diff_arr,diff_arr)))

'''
mode = int(input('Mean, Median, Gaussian, Bilateral : 1,2,3,4 : '))
test case : 
    kernel size = 5, 7
    noise level = 15, 25, 50
'''
mode = '3'
kernel_size = 5
window_size = 21

mean = 0
sigma_noise = 20
sigma_gauss_filter = 10
sigma_range = 30

# cv reads image in BGR format
BGR_img = cv2.imread('Lena.png')
img_gray = cv2.cvtColor(BGR_img, cv2.COLOR_BGR2GRAY)

gaussian =  np.random.normal(mean, sigma_noise, img_gray.shape) 
noisy_img = np.zeros(img_gray.shape, np.float32)
noisy_img[:,:] = img_gray[:,:] + gaussian
#cv2.normalize(noisy_img, noisy_img, 0, 255, cv2.NORM_MINMAX, dtype=-1)

npad_gray = [(int((kernel_size-1)/2),int((kernel_size-1)/2)),(int((kernel_size-1)/2),int((kernel_size-1)/2))]
img_gray_padded = np.pad(noisy_img, npad_gray, mode='constant')

npad_gray_nlm = [(int((window_size-1)/2),int((window_size-1)/2)),(int((window_size-1)/2),int((window_size-1)/2))]
img_gray_padded_nlm = np.pad(noisy_img, npad_gray_nlm, mode='constant')

filtered_img_gray = np.zeros(img_gray.shape)

###############################################################################
'''
filter_param = sigma_noise*10
search_window = np.zeros([window_size,window_size])
weight_a = np.zeros([window_size-6,window_size-6])
weight_sum = 0
filtered_pix = 0

def multi_loop_nlm(tup_ij):
    i = tup_ij[0]
    j = tup_ij[1]
    filtered_pix = 0
    weight_sum = 0
    
    temp = img_gray_padded_nlm[i:i+window_size,j:j+window_size]
    center_kernel = temp[7:14,7:14]

    for k,l in np.ndindex((window_size-6,window_size-6)):
        temp_kernel = temp[k:k+7,l:l+7]
        #val = math.exp(-euclidean_distance(center_kernel,temp_kernel)/filter_param**2)
        val = math.exp(-np.linalg.norm(center_kernel-temp_kernel)/filter_param**2)
        weight_a.itemset((k,l),val)
        filtered_pix += val*temp_kernel.item(3,3)

    weight_sum = np.sum(weight_a)
    filtered_pix /= weight_sum
    return (i,j,filtered_pix)       
    #filtered_img_gray.itemset((i,j),filtered_pix)
'''
###############################################################################

if mode == '1':
    # Mean (averaging)
    kernel = np.full((kernel_size,kernel_size),1/(kernel_size**2))
    for i in range(0,img_gray.shape[0]):
        for j in range(0,img_gray.shape[1]):
            temp = img_gray_padded[i:i+kernel_size,j:j+kernel_size]
            filtered_img_gray.itemset((i,j),sum(sum(np.multiply(kernel,temp))))

elif mode == '2':
    # Median
    for i in range(0,img_gray.shape[0]):
        for j in range(0,img_gray.shape[1]):
            temp = sorted(list(img_gray_padded[i:i+kernel_size,j:j+kernel_size].flat))
            filtered_img_gray.itemset((i,j),temp[int((kernel_size**2 - 1)/2)])        
            
elif mode == '3':
    # Gaussian
    kernel = np.zeros([kernel_size,kernel_size])
    for i in range(0,kernel_size):
        for j in range(0,kernel_size):
            kernel.itemset((i,j),(-(i-int((kernel_size-1)/2))**2-(j-int((kernel_size-1)/2))**2)/(2*sigma_gauss_filter**2))
    kernel = np.exp(kernel)
    kernel = kernel/sum(sum(kernel))
   
    for i in range(0,img_gray.shape[0]):
        for j in range(0,img_gray.shape[1]):
            temp = img_gray_padded[i:i+kernel_size,j:j+kernel_size]
            filtered_img_gray.itemset((i,j),sum(sum(np.multiply(kernel,temp))))

elif mode == '4':
    # Bilateral Filter
    kernel = np.zeros([kernel_size,kernel_size])
    kernel_spatial = np.zeros([kernel_size,kernel_size])
    kernel_range = np.zeros([kernel_size,kernel_size])
    for i in range(0,kernel_size):
        for j in range(0,kernel_size):
            kernel_spatial.itemset((i,j),(-(i-int((kernel_size-1)/2))**2-(j-int((kernel_size-1)/2))**2)/(2*sigma_gauss_filter**2))

    for i in range(0,img_gray.shape[0]):
        for j in range(0,img_gray.shape[1]):
            temp = img_gray_padded[i:i+kernel_size,j:j+kernel_size]
            
            for k in range(0,kernel_size):
                for l in range(0,kernel_size):
                    kernel_range.itemset((k,l),(-(img_gray_padded.item(i+k,j+l)-noisy_img.item(i,j))**2)/(2*sigma_range**2))
                    
            kernel = np.add(kernel_spatial,kernel_range)
            kernel = np.exp(kernel)
            kernel = kernel/sum(sum(kernel))
   
            filtered_img_gray.itemset((i,j),sum(sum(np.multiply(kernel,temp))))

elif mode == '5':
    # Non-Local Means
    
    #filter_param = sigma_noise*10
    filter_param = 10
    search_window = np.zeros([window_size,window_size])
    weight_a = np.zeros([window_size-6,window_size-6])
    weight_sum = 0
    filtered_pix = 0

    for i,j in np.ndindex(img_gray.shape):

        filtered_pix = 0
        weight_sum = 0
        
        temp = img_gray_padded_nlm[i:i+window_size,j:j+window_size]
        center_kernel = temp[7:14,7:14]

        for k,l in np.ndindex((window_size-6,window_size-6)):
            temp_kernel = temp[k:k+7,l:l+7]
            #val = math.exp(-euclidean_distance(center_kernel,temp_kernel)/filter_param**2)
            val = math.exp(-np.linalg.norm(center_kernel-temp_kernel)/filter_param**2)
            weight_a.itemset((k,l),val)
            filtered_pix += val*temp_kernel.item(3,3)
            
        weight_sum = np.sum(weight_a)
        filtered_pix /= weight_sum
            
        filtered_img_gray.itemset((i,j),filtered_pix)
    
# 1 : clean vs noisy
# 2 : clean vs filtered
PSNR1 = psnr(img_gray,noisy_img)
PSNR2 = psnr(img_gray,filtered_img_gray)

if kernel_size == 5:
    plt.subplot(131),plt.imshow(img_gray, cmap=plt.cm.gray),plt.title('Original')
    plt.xticks([]), plt.yticks([])
    plt.subplot(132),plt.imshow(noisy_img, cmap=plt.cm.gray),plt.title('Noise Added')
    plt.xticks([]), plt.yticks([])
    plt.subplot(133),plt.imshow(filtered_img_gray, cmap=plt.cm.gray),plt.title('Filtered')
    plt.xticks([]), plt.yticks([])
    plt.show()

else:
    plt.subplot(133),plt.imshow(filtered_img_gray, cmap=plt.cm.gray),plt.title('Filtered')
    plt.xticks([]), plt.yticks([])
    plt.show()

print(PSNR1)
print(PSNR2)