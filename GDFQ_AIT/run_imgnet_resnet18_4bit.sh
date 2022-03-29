#!/bin/bash

python main.py --conf_path ./imagenet_resnet18.hocon --qw 4 --qa 4 --id 01 --ce_scale 0.0 --kd_scale 1.0 --lrs 0.0001 --adalr