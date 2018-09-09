import cv2
import sys

name = sys.argv[1]

im = cv2.imread(name+'.png')
gray_im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
thresh, im_bw = cv2.threshold(gray_im, 190, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

cv2.imwrite(name+'_binary.png', im_bw)