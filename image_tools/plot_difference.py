import cv2
import sys
import numpy as np
from PIL import Image
from PIL import ImageChops

#im1 = cv2.imread(sys.argv[1])
#im2 = cv2.imread(sys.argv[2])
im1 = Image.open(sys.argv[1]).convert('L')
im2 = Image.open(sys.argv[2]).convert('L')

diff = ImageChops.difference(im1, im2)

diff.save(sys.argv[3])
#im = np.abs(im2-im1)

#cv2.imwrite('res.png', im)
