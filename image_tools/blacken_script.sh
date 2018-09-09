!/bin/bash
image=100
while [ $image -le 135 ]
do
    echo /home/zoro/pytorch-CycleGAN-and-pix2pix/cards_first_stage_model/test_latest/images/"${image}"_real_B
    python blacken.py /home/zoro/pytorch-CycleGAN-and-pix2pix/cards_first_stage_model/test_latest/images/"${image}"_real_B 60 270 665 335
    ((image++))
done
