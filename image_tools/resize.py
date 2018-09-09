import cv2
import sys

image_name = sys.argv[1]

im = cv2.imread(image_name)
im = cv2.resize(im, (1024, 820))

cv2.imwrite(image_name, im)