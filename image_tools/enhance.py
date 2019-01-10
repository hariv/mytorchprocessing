import cv2
import sys
import numpy as np
from PIL import Image
from PIL import ImageChops

im = cv2.imread(sys.argv[1])

print("Working on "+sys.argv[1])
height, width, channels = im.shape

for i in range(height):
    for j in range(width):
        for k in range(channels):
            if(im[i,j,k]*50 > 255):
                im[i,j,k] = 255
            else:
                im[i,j,k] = im[i,j,k] * 50

cv2.imwrite(sys.argv[2], im)
