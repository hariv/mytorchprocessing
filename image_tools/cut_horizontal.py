import cv2
import sys

fileCount = 1

y_start = 0
y_end = 618
for i in range(1, 16):
    im = cv2.imread(str(i)+'_1.png')
    print(str(i)+'_1.png')
    image = im[y_start:y_end,:,:]
    cv2.imwrite(str(fileCount)+'.png', image)
    fileCount = fileCount + 3