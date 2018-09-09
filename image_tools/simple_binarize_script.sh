!/bin/bash
image=1
while [ $image -le 12 ]
do
    python simple_binarize.py "${image}_digits"
    ((image++))
done
