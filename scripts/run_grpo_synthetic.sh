#!/bin/bash

PYTHONPATH=$PWD/src ACCELERATE_LOG_LEVEL=info \
    accelerate launch --config_file recipes/accelerate_configs/zero3.yaml \
    src/open_r1/grpo_synthetic.py --config recipes/Qwen3/grpo/config_synthetic.yaml

# gcloud storage cp -r /tmp/synthetic_data_generator_Qwen3-1.7B_step1 gs://cmu-gpucloud-zichunyu/out