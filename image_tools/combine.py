import cv2
import sys

image_name = sys.argv[1]
mask_first_name = sys.argv[2]
mask_last_name = sys.argv[3]
mask_digits_name = sys.argv[4]

first_start_x = sys.argv[5]
first_start_y = sys.argv[6]
first_end_x = sys.argv[7]
first_end_y = sys.argv[8]

last_start_x = sys.argv[9]
last_start_y = sys.argv[10]
last_end_x = sys.argv[11]
last_end_y = sys.argv[12]

digits_start_x = sys.argv[13]
digits_start_y = sys.argv[14]
digits_end_x = sys.argv[15]
digits_end_y = sys.argv[16]

image = cv2.imread(image_name+'.png')
mask_first = cv2.imread(mask_first_name+'.png')
mask_last = cv2.imread(mask_last_name+'.png')
mask_digits = cv2.imread(mask_digits_name+'.png')

image[int(first_start_y):int(first_end_y), int(first_start_x):int(first_end_x)] = mask_first
image[int(last_start_y):int(last_end_y), int(last_start_x):int(last_end_x)] = mask_last
image[int(digits_start_y):int(digits_end_y), int(digits_start_x):int(digits_end_x)] = mask_digits

cv2.imwrite(image_name+'_train_temp.png', image)