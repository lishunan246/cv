import numpy as np
import cv2
from matplotlib import pyplot as plt
from scipy import signal

img=cv2.imread('g1.jpg',0)
img2=cv2.imread('g2.jpg',0)
imgx=cv2.imread('gx.jpg',0)

plt.subplot(221),plt.imshow(img,cmap='gray')
plt.title('1'), plt.xticks([]), plt.yticks([])
plt.subplot(222),plt.imshow(img2,cmap='gray')
plt.title('2'), plt.xticks([]), plt.yticks([])
plt.subplot(223),plt.imshow(imgx,cmap='gray')
plt.title('x'), plt.xticks([]), plt.yticks([])

conv1=signal.fftconvolve(img,cv2.flip(img, -1),mode='same')
conv2=signal.fftconvolve(imgx,cv2.flip(imgx, -1),mode='same')
conv3=signal.fftconvolve(img2,cv2.flip(img2, -1),mode='same')
conv=signal.fftconvolve(img,cv2.flip(img2, -1),mode='same')
convx=signal.fftconvolve(img,cv2.flip(imgx, -1),mode='same')
convx2=signal.fftconvolve(img2,cv2.flip(imgx, -1),mode='same')

print '1-1', conv1.max()
print 'x-x', conv2.max()
print '2-2', conv3.max()
print '1-2', conv.max()
print '1-x', convx.max()
print '2-x', convx2.max()
plt.show()
