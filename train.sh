#!/bin/bash

python train.py --config=./configs/nasnet_mobile.yml --fold=3 --batch_size=64 --lr=0.0001
