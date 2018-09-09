import cv2 as cv
import numpy as np
#from matplotlib import pyplot as plt
img = cv.imread('noisy2.png',0)
# global thresholding
th1, ret1 = cv.threshold(img,127,255,cv.THRESH_BINARY)
# Otsu's thresholding
th2,ret2 = cv.threshold(img,0,255,cv.THRESH_BINARY+cv.THRESH_OTSU)
# Otsu's thresholding after Gaussian filtering
blur = cv.GaussianBlur(img,(5,5),0)
th3,ret3 = cv.threshold(blur,0,255,cv.THRESH_BINARY+cv.THRESH_OTSU)

cv.imwrite('hari_otsu_global.png', ret1)
cv.imwrite('hari_otsu_otsu.png', ret2)
cv.imwrite('hari_otsu_gaussian.png', ret3)
