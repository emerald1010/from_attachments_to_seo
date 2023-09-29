# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
#!/bin/bash

DIR="../dataset_subset_root"
ARCH="alexnet"
LR=0.05
WD=-5
K=900
WORKERS=20
EXP="../exp_folder"
PYTHON="/home/<USER>/miniconda3/envs/deepcluster/bin/python3.7"

mkdir -p ${EXP}

CUDA_VISIBLE_DEVICES=0 ${PYTHON} main2.py ${DIR} --exp ${EXP} --arch ${ARCH} \
  --lr ${LR} --wd ${WD} --k ${K} --verbose --workers ${WORKERS} --sampler --epochs 1000
