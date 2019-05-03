#!/bin/bash

FOLD=3

python inference.py --config=./configs/resnet50.$FOLD.yml \
					--save_probs=./savings/resnet50_fold$FOLD/probs.csv \
					--threshold=0.1