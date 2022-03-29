#!/bin/bash

python main.py --conf_path ./imagenet_resnet50.hocon --multi_label_prob 0.4 --multi_label_num 100 --id 01 --qw 4 --qa 4 --ce_scale 0.0 --kd_scale 1.0 --adalr 