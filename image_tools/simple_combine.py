import cv2
import sys

image_name = sys.argv[1]
mask_name = sys.argv[2]
start_x = sys.argv[3]
start_y = sys.argv[4]
end_x = sys.argv[5]
end_y = sys.argv[6]

image = cv2.imread(image_name+'.png')
mask = cv2.imread(mask_name+'.png')

image[int(start_y):int(end_y), int(start_x):int(end_x)] = mask

cv2.imwrite(image_name+'_train_temp.png', image)