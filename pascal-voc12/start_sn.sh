#!/bin/bash

# Start training on a single node

set -xeu

echo "Update dependencies"
pip install --upgrade pytorch-ignite
pip install fire py-config-runner git+https://github.com/vfdev-5/ImageDatasetViz.git albumentations opencv-python-headless clearml


echo "Setup ClearML logging"
export CLEARML_API_HOST="https://api.community.clear.ml"
export CLEARML_WEB_HOST="https://app.community.clear.ml"
export CLEARML_FILES_HOST="https://files.community.clear.ml"


echo "Start training"
export DATASET_PATH=/opt/trainml/input/
export SBD_DATASET_PATH=/opt/trainml/input/VOCaug/dataset/

python -u -m torch.distributed.launch --nproc_per_node=$1 --use_env main.py training configs/baseline_dplv3_resnet101_sbd.py
