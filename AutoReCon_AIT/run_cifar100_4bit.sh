#!/bin/bash

python main.py --conf_path ./cifar100_resnet20.hocon --qw 4 --qa 4 --id 01 --ce_scale 0.0 --kd_scale 1.0 --lrs 0.0001 --passing_threshold 0.001 --adalr