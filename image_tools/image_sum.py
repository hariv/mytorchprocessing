import cv2
import sys
import numpy as np
from PIL import Image
from PIL import ImageChops

im = cv2.imread(sys.argv[1])
#channel = int(sys.argv[2])
print(np.sum(im))
#im1 = cv2.imread(sys.argv[1])
#im2 = cv2.imread(sys.argv[2])
#im1 = Image.open(sys.argv[1]).convert('RGB')
#im2 = Image.open(sys.argv[2]).convert('RGB')

#im1 = np.uint8(im1)
#im2 = np.uint8(im2)
#im1 = im1.tolist()
#im2 = im2.tolist()

#diff =  map(abs, im1-im2)

#diff = np.zeros(im1.shape)
#diff = np.abs(im1.astype("float")-im2.astype("float"))
#diff = np.abs(im1-im2)
#diff = np.abs(im2-im1)
#diff = ImageChops.difference(im1, im2)

#cv2.imwrite(sys.argv[3], diff)
#diff.save('res.png')
#diff.save(sys.argv[3])
#im = np.abs(im2-im1)

#cv2.imwrite('res.png', np.asarray(diff))
