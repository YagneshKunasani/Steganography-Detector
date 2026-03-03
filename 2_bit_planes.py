import cv2
import numpy as np
import matplotlib.pyplot as plt

img_original = cv2.imread('clean_0.png', 1) 
img_stego = cv2.imread('stego_0.png', 1)

blue_original = img_original[:, :, 0]
blue_stego = img_stego[:, :, 0]

plt.figure(figsize=(20,10))


plt.subplot(2,2,1)
plt.imshow(blue_original, cmap='gray')
plt.title('Original Blue Channel')

plt.subplot(2,2,2)
plane_0 = (blue_original >> 0) & 1
plt.imshow(plane_0 * 255, cmap='gray')
plt.title('Original LSB (Random Noise)')


plt.subplot(2,2,3)
plt.imshow(blue_stego, cmap='gray')
plt.title('Stego Blue Channel (Looks Normal)')

plt.subplot(2,2,4)
plane_0_new = (blue_stego >> 0) & 1
plt.imshow(plane_0_new * 255, cmap='gray')
plt.title('Stego LSB (THE HIDDEN PATTERN)')

plt.show()