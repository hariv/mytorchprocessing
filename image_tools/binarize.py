import cv2
import sys

name = sys.argv[1]

im = cv2.imread(name+'.png')
gray_im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
#thresh, im_bw = cv2.threshold(gray_im, 128, 255, cv2.THRESH_BINARY)                                                                                                                                                

img = cv2.medianBlur(gray_im,5)
ret,th1 = cv2.threshold(img,127,255,cv2.THRESH_BINARY)
th2 = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,11,2)
th3 = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)

#cv2.imwrite('th1.png', th1)                                                                                                                                                                                        
cv2.imwrite(name+'_binary.png', 255-th2)
#cv2.imwrite('th3.png', th3)                                                                                                                                                                                        
#cv2.imwrite(name+'_binary.png', 255-im_bw)