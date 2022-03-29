#!/bin/bash

python main.py --conf_path ./imagenet_mobilenet_v2.hocon --multi_label_prob 0.4 --multi_label_num 250 --id 01 --randemb --qw 4 --qa 4 --ce_scale 0.0 --kd_scale 1.0 --adalr