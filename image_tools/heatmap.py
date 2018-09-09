import cv2
import numpy as np

im = cv2.imread('../nonoise-lowres-fixed.png')
height, width, channels = im.shape

x = 23
y = 20

x_count = 1
y_count = 1

while y_count < 78:
	
	lines = [line.rstrip('\n') for line in open(str(y_count)+'.txt')]

	for i in range(len(lines)):
		start_x = x
		start_y = y
		end_x = start_x+9
		end_y = start_y+9

		if(float(lines[i]) >= 0.9):
			im[start_y:end_y, start_x:end_x, 0] = 0
			im[start_y:end_y, start_x:end_x, 1] = 0
			im[start_y:end_y, start_x:end_x, 2] = 255
		elif(float(lines[i]) >= 0.8 and float(lines[i]) < 0.9):
			im[start_y:end_y, start_x:end_x, 0] = 0
			im[start_y:end_y, start_x:end_x, 1] = 0
			im[start_y:end_y, start_x:end_x, 2] = 127
		elif(float(lines[i]) >= 0.7 and float(lines[i]) < 0.8):
			im[start_y:end_y, start_x:end_x, 0] = 0
			im[start_y:end_y, start_x:end_x, 1] = 0
			im[start_y:end_y, start_x:end_x, 2] = 63
		elif(float(lines[i]) >= 0.6 and float(lines[i]) < 0.7):
			im[start_y:end_y, start_x:end_x, 0] = 0
			im[start_y:end_y, start_x:end_x, 1] = 200
			im[start_y:end_y, start_x:end_x, 2] = 0
		elif(float(lines[i]) >= 0.5 and float(lines[i]) < 0.6):
			im[start_y:end_y, start_x:end_x, 0] = 0
			im[start_y:end_y, start_x:end_x, 1] = 127
			im[start_y:end_y, start_x:end_x, 2] = 0
		elif(float(lines[i]) >= 0.4 and float(lines[i]) < 0.5):
			im[start_y:end_y, start_x:end_x, 0] = 0
			im[start_y:end_y, start_x:end_x, 1] = 63
			im[start_y:end_y, start_x:end_x, 2] = 0
		elif(float(lines[i]) >= 0.3 and float(lines[i]) < 0.4):
			im[start_y:end_y, start_x:end_x, 0] = 255
			im[start_y:end_y, start_x:end_x, 1] = 0
			im[start_y:end_y, start_x:end_x, 2] = 0
		elif(float(lines[i]) >= 0.2 and float(lines[i]) < 0.3):
			im[start_y:end_y, start_x:end_x, 0] = 127
			im[start_y:end_y, start_x:end_x, 1] = 0
			im[start_y:end_y, start_x:end_x, 2] = 0
		elif(float(lines[i]) >= 0.1 and float(lines[i]) < 0.2):
			im[start_y:end_y, start_x:end_x, 0] = 63
			im[start_y:end_y, start_x:end_x, 1] = 0
			im[start_y:end_y, start_x:end_x, 2] = 0
		else:
			im[start_y:end_y, start_x:end_x, 0] = 0
			im[start_y:end_y, start_x:end_x, 1] = 0
			im[start_y:end_y, start_x:end_x, 2] = 0

		x_count += 1
		x = end_x + 1

	y_count += 1
	x_count = 1
	x = 23
	y = end_y + 1
	cv2.imwrite('./heat.png', im)