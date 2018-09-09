import cv2
import sys

name = sys.argv[1]
start_x = sys.argv[2]
start_y = sys.argv[3]
end_x = sys.argv[4]
end_y = sys.argv[5]

im = cv2.imread(name+'.png')
im[int(start_y):int(end_y),int(start_x):int(end_x), :] = 0

cv2.imwrite(name+'_black.png', im)
