import cv2
import sys

name = sys.argv[1]
horizontal_size = int(sys.argv[2])
vertical_size = int(sys.argv[3])

horizontal_limit = 3
vertical_limit = 1

count = 1

im = cv2.imread(name)

for i in range(0, vertical_limit):
	for j in range(0, horizontal_limit):
		image = im[i*vertical_size:(i+1)*vertical_size, 285+(j*horizontal_size):285+((j+1)*horizontal_size), :]
		cv2.imwrite(str(count)+'.png', image)
		count += 1
