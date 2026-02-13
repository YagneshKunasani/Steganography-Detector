import cv2
import numpy as np

image = cv2.imread("test_image.jpg")

print(f"Image shape is: {image.shape}")

pixel_value = image[0,0]
print(f"Pixel at (0,0) contains these colors (B,G,R): {pixel_value}")

red_value = pixel_value[2]
print(red_value)

print(f"Binary representation of the red value of 1st pixel: {format(red_value,'08b')}")