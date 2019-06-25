#! /usr/bin/env python
import os
import json
from yolo_ops import make_train_test_data
from yolo_v2.train_ops import train

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"

config_path = './config.json'
with open(config_path) as config_buffer:
    config = json.loads(config_buffer.read())

if __name__ == '__main__':
    make_train_test_data(make_data=True)
    train()

