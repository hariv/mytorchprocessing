import cv2
import sys

fileCount = 10
y_start = 0
y_end = 2528

for i in range(366, 372):
    im = cv2.imread('IMG_0'+str(i)+'.jpg')
    print('IMG_0'+str(i)+'.jpg')
    x_start = 277
    x_end = 1288

    for j in range(1, 4):
        image = im[y_start:y_end, x_start:x_end, :]
        cv2.imwrite(str(fileCount)+"_"+str(j)+'.png', image)
        x_start = x_end
        x_end = x_start + 1011
    fileCount = fileCount + 1