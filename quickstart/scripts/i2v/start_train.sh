#! /usr/bin/env bash

torchrun \
    --nproc_per_node=1 \
    --master_port=29501 \
    ../train.py \
    --yaml config.yaml
