#!/bin/sh
#cd ..

python train_model.py -d=mnist -e=10
python craft_adv_samples.py -d=mnist -a=fgsm
python detect_adv_samples.py -d=mnist -a=fgsm