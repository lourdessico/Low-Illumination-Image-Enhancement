# -*- coding: utf-8 -*-
"""
Created on Fri Jul  5 17:38:06 2024

@author: lourd
"""

# PAPER PROJECT 2024
# ORIOL JOSA & LOURDES SIMON

# Nighttime low illumination image enhancement with single image using
# bright/dark channel prior
# Licensed under the MIT License.

import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage

# The code is structured to apply the method on one image at a time.
# Here we wrote de code to import the images we have worked with

im= plt.imread("plant.jpg")
# im= plt.imread("camping.jpg")
# im= plt.imread("laboratory.jpg")
# im= plt.imread("statue.jpg")
# im= plt.imread("light.jpg")


# "im" must be RGB image (3 channels, not 4) and normalized
if np.max(im) > 1:
    im = im / 255
if im.ndim < 3:
    print("The image is not RGB")
if im.shape[2] == 4:
    im = im[:,:,:3]

rows = np.shape(im)[0]
columns = np.shape(im)[1]

# Plot the original image
plt.figure()

plt.title('Original image')
plt.axis("off")
plt.imshow(im)

plt.show()


#%%


# 1- Dark/bright channel prior

# Firstly, we use a max filter and a min filter to obtain the bright 
#channel and the dark channel of input image, respectively
def bright_ch(im, path):
    bright = np.ones_like(im, dtype = np.float64)
    maxR = ndimage.maximum_filter(im[:,:,0], size = path)
    maxG = ndimage.maximum_filter(im[:,:,1], size = path)
    maxB = ndimage.maximum_filter(im[:,:,2], size = path)

    bright = np.maximum(np.maximum(maxR, maxG), maxB)
    return bright

def dark_ch(im, path):
    dark = np.ones_like(im, dtype = np.float64)
    minR = ndimage.minimum_filter(im[:,:,0], size = path)
    minG = ndimage.minimum_filter(im[:,:,1], size = path)
    minB = ndimage.minimum_filter(im[:,:,2], size = path)

    dark = np.minimum(np.minimum(minR, minG), minB)
    return dark


# Set the dimensions of the local patch, depending on the image size
patch = (15,15)

# Get the dark and bright channels of our image using our functions
dark = dark_ch(im, patch)
bright = bright_ch(im, patch)


# Plot the dark and bright channels
plt.figure()

plt.subplot(2, 1, 1)
plt.title('Dark channel')
plt.axis("off")
plt.imshow(dark, cmap = "gray")

plt.subplot(2, 1, 2)
plt.title('Bright channel')
plt.axis("off")
plt.imshow(bright, cmap = "gray")

plt.tight_layout()
plt.show()


#%%


# 2- Atmospheric light estimation

# Function that calculates the atmospheric light estimation of a channel, using
# the bright channel and the channel of the image that we choose
def Atmospheric_light(channel, bright):
    # Take the 10% of the bright channel pixels (brightest pixels)
    npixels = int(0.1*channel.size)

    # To make it easier we will work with 1D arrays
    # We get a flattened array with .ravel()
    bright_1D = bright.ravel()
    channel_1D = channel.ravel()
    
    # Using np.argsort() we get a 1-D array containing the index of the elements
    # ordered from lowest to highest

    # We only need the 10% brightest pixels, therefore we use the last npixels
    positions = np.argsort(bright_1D)[bright_1D.size-npixels:]


    # Calculate the value of the atmospheric light corresponding to the channel
    # chosen by calculating the mean of the channel values that correspond to positions  
    A = np.mean(channel_1D[positions])

    return A


# Calculate the atmospheric light estimation of each channel
AR = Atmospheric_light(im[:,:,0], bright)
AG = Atmospheric_light(im[:,:,1], bright)
AB = Atmospheric_light(im[:,:,2], bright)

# Make an array with the values to operate more easily afterwards
A = np.array([AR,AG,AB])


#%%


# 3- Estimating the transmission using dark/bright channel prior

# Functions that calculate the transmission based on the bright or dark channels
# according to the paper indications
def t_bright(A, bright):
    t = (bright-np.max(A))/(1-np.max(A))
    return t


def t_dark(A, dark):
    t = 1-(dark/np.min(A))

    return t


# Calculate the transmissions using the functions defined previously
t_bright = t_bright(A, bright)
t_dark = t_dark(A, dark)

# Calculate the difference between the bright and dark channel
difference = (bright-dark)

# Threshold determined by empirical experiment
threshold = 0.4

# Calculate the corrected transmission by multiplying the dark and bright transmissions
t_corr = t_bright*t_dark

# To calculate the final transmission we take the value of the corrected
# transmission when the difference between the bright and dark channel
# is lower than the threshold, otherwise the pixel takes the t_bright value
transmission = np.ones_like(dark, dtype = np.float64)
transmission = np.where((difference)<threshold, t_corr, t_bright)


# Plot the 4 different transmissions
plt.figure()
plt.subplot(2, 2, 1)
plt.title('Bright transmission')
plt.axis("off")
plt.imshow(t_bright, cmap = "gray")

plt.subplot(2, 2, 2)
plt.title('Dark transmission')
plt.axis("off")
plt.imshow(t_dark, cmap = "gray")

plt.subplot(2, 2, 3)
plt.title('Corrected transmission')
plt.axis("off")
plt.imshow(t_corr, cmap = "gray")

plt.subplot(2, 2, 4)
plt.title('Transmission')
plt.axis("off")
plt.imshow(transmission, cmap = "gray")

plt.tight_layout()
plt.show()


#%%


# 4- Transmission map refinement using guided filter

def guided_filter(I, p, eps, nw):
    
    n, m = p.shape # Associate variables to the dimensions of p

    # To calculate the mean filter of an array we use ndimage.uniform_filter()
    
    # Calculate the mean of the different channels of I 
    mean_R = ndimage.uniform_filter(I[:,:,0], size = nw)
    mean_G = ndimage.uniform_filter(I[:,:,1], size = nw)
    mean_B = ndimage.uniform_filter(I[:,:,2], size = nw)

    # Calculate the mean of p
    mean_p = ndimage.uniform_filter(p, size = nw)

    # Calculate the mean of I*p for each channel
    mean_Rp = ndimage.uniform_filter(I[:,:,0]*p, size = nw)
    mean_Gp = ndimage.uniform_filter(I[:,:,1]*p, size = nw)
    mean_Bp = ndimage.uniform_filter(I[:,:,2]*p, size = nw)

    # Calculate the covariance between the different channels of I and p
    cov_Rp = mean_Rp - mean_R * mean_p
    cov_Gp = mean_Gp - mean_G * mean_p
    cov_Bp = mean_Bp - mean_B * mean_p

    # Calculate the covariance between all the possible combinations of the channels of I
    cov_RR = ndimage.uniform_filter(I[:,:,0]*I[:,:,0], size = nw) - mean_R * mean_R
    cov_RG = ndimage.uniform_filter(I[:,:,0]*I[:,:,1], size = nw) - mean_R * mean_G
    cov_RB = ndimage.uniform_filter(I[:,:,0]*I[:,:,2], size = nw) - mean_R * mean_B
    cov_GG = ndimage.uniform_filter(I[:,:,1]*I[:,:,1], size = nw) - mean_G * mean_G
    cov_GB = ndimage.uniform_filter(I[:,:,1]*I[:,:,2], size = nw) - mean_G * mean_B
    cov_BB = ndimage.uniform_filter(I[:,:,2]*I[:,:,2], size = nw) - mean_B * mean_B

    # Calculate the coefficients ak using a 3-D matrix
    ak = np.zeros((n, m, 3))
    
    # Create a covariance matrix between the RGB channels (n, m, 3, 3)
    sigma = np.stack([np.stack([cov_RR, cov_RG, cov_RB], axis=-1),
                      np.stack([cov_RG, cov_GG, cov_GB], axis=-1),
                      np.stack([cov_RB, cov_GB, cov_BB], axis=-1)], axis=-2)
    
    # Create a matrix of the covariance between p and each channels 
    # and add a dimention (n, m, 3, 1) to operate easily
    covIp = (np.stack([cov_Rp, cov_Gp, cov_Bp], axis=-1))[:, :, :, np.newaxis]
            
    # Reshhape the identity matrix to (1, 1, 3, 3) to operate
    eye = np.eye(3).reshape(1, 1, 3, 3)
    
    #Add to sigma
    sigma += eps * eye

    # Solve the linear matrix equation for ak (eq.(19)) and remove the last axis
    ak = np.linalg.solve(sigma, covIp).squeeze(-1)
    
    # Calculate the coefficient bk matrix (eq.(20))
    bk = mean_p - ak[:,:,0] * mean_R - ak[:,:,1] * mean_G - ak[:,:,2] * mean_B

    # Calculate the mean of the coefficient matrixes ak and bk 
    meanA = np.zeros((n, m, 3))
    meanA[:,:,0] = ndimage.uniform_filter(ak[:,:,0], size = nw)
    meanA[:,:,1] = ndimage.uniform_filter(ak[:,:,1], size = nw)
    meanA[:,:,2] = ndimage.uniform_filter(ak[:,:,2], size = nw)
    
    
    meanB = ndimage.uniform_filter(bk, size = nw)

    # Calulate the resulting matrix (eq.(18))
    q = np.sum(meanA * I, axis=2) + meanB

    return q


eps = 0.001 # Give a value to eps, a regularization parameter
nw = 50 # Give a value to the side size of the neighbourhood

t = np.zeros_like(im)

# Call the function using "transmission" as "p" and "im" as "I" to get the
# final transmission
t = guided_filter(im, transmission, eps, nw)

# In order to avoid errors the paper reccomends to use the following parameter 
# that restricts the transmission t(x) to a lower bound t_0
# We transformed t_0 and t into 3-D matrices to operate
to = 0.1 * np.ones(im.shape)
t_3D= np.stack([t] * 3, axis=2)

# To operate we calculate a 3-D matrix the same size as the image where in each
# channel all the pixels have the same value, the one corresponding to it's 
# atmospheric light "A^c"
A_matrix = np.ones_like(im, dtype=np.float64)*A

# To calculate the final enhanced image we use the equation given in the paper 
Final = np.ones_like(im, dtype=np.float64) 
Final = ((im-A_matrix)/np.maximum(t_3D, to)) + A_matrix

# Redistribute the information of the matrix between 0 and 1 so it
# can be represented
Final = (Final - np.min(Final))
Final = Final/np.max(Final)


# Plot the transmission and the final enhanced image
plt.figure()

plt.title('t(x) after guided filter')
plt.axis("off")
plt.imshow(t, cmap = "gray")

plt.show()

plt.figure()

plt.title('Final image')
plt.axis("off")
plt.imshow(Final)

plt.show()

# modify the name or file type when saving the image
# plt.imsave('plant_results.jpg', Final)



#%%

# ANNEX

# Alternative ways to do some procedures that were discarted as we found a way
# which provided us with better or very similar results.


# Alternative function that calculates the transmission based on the dark
# channels according to the eq. 11 from the paper [3]
# def t_dark(dark_norm):
#     t = 1-dark_norm
#     return t

# We divide each channel by it's corresponding value of A:
# im_normA = np.ones_like(im, dtype=np.float64)
# for i in range(3):
#   im_normA[:,:,i]=im[:,:,i]/A[i]

# Then we calculate this new im_normA dark channel:
# dark_norm = dark_ch(im_normA, path)
# t_dark = t_dark(dark_norm)

# Using this way we obtain really similar results but we stayed with the other
# one because it was the most coherent with our paper and how we calculate t_bright

# ___________________________________________________________________________________

# Alternative way to calculate the mean of the coefficient matrix ak in the guided
# filter. Instead of calculating the mean of each channel, we can calculate the mean
# of the entire matrix (which also mixes the channels when calculating the mean):

# meanA = ndimage.uniform_filter(ak, size = nw)

# ___________________________________________________________________________________

# Alternative way of the use of guided_filter. In this case both p and I are 2D images:

# def guided_filter(I, p, eps, nw):

#     mean_p = ndimage.uniform_filter(p, size = nw)
#     mean_I = ndimage.uniform_filter(I, size = nw)
#     var_I= ndimage.uniform_filter(I**2, size = nw) - mean_I**2
#     mean_Ip = ndimage.uniform_filter(p*I, size = nw)
#     cov_Ip = mean_Ip - mean_p * mean_I

#     ak= cov_Ip/(var_I+eps)
#     bk= mean_p - ak* mean_I

#     meanA = ndimage.uniform_filter(ak, size = nw)
#     meanB = ndimage.uniform_filter(bk, size = nw)

#     q = meanA*I+meanB

#     return q


# Then when call the function we use "transmission" as "I" and "Luma" as "p"
# This way we are faithful to the equation (21).

# t= guided_filter(transmission, Luma, eps, nw)