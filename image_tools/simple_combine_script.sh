!/bin/bash
image=1
while [ $image -le 12 ]
do
    #echo "${image}" "${image}"_digits_binary 70 280 650 330
    python simple_combine.py "${image}" "${image}"_digits_binary 70 280 650 330
    ((image++))
done



