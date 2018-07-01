#!/bin/bash
im=1
while [ $im -le 7 ]
do
    cp ~/"${im}".jpg ~/service/data/val/sign
    python main.py -a alexnet --pretrained --evaluate /home/zoro/service/data
    rm ~/service/data/val/sign/*
    ((im++))
done
echo All Done
