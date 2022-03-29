#!/bin/bash

python main.py --conf_path ./cifar10_resnet20.hocon --multi_label_prob 0.4 --multi_label_num 2 --id 01 --qw 4 --qa 4 --ce_scale 0.0 --kd_scale 1.0 --passing_threshold 0.001 --adalr 