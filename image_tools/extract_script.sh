!/bin/bash
image=1
while [ $image -le 12 ]
do
    python extract.py "${image}" 70 280 650 330
    ((image++))
done
