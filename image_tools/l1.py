import cv2
import numpy as np
import sys
import glob

start = sys.argv[1]
end = sys.argv[2]

base_location = '~/pytorch-CycleGAN-and-pix2pix/results/cards_digits_autoencoder/test_latest/images/'
real_extension = '_digits_real_A.png'
generated_extension = '_digits_fake_B.png'

for i in range(int(start), int(end)):
    real_file = cv2.imread(base_location+str(i)+real_extension)
    generated_file = cv2.imread(base_location+str(i)+generated_extension)
    print(np.sum(np.abs(real_file-generated_file)))