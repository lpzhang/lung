#!/bin/bash
PYTHONPATH=/home/zlp/dev/caffe/python:$PYTHONPATH
#nohup python -u ~/dev/medseg/tools/train_net.py --gpu=0  >& train.log 2>&1 &
python -u ~/dev/lung/main.py --gpu=0  2>&1 | tee test.log &