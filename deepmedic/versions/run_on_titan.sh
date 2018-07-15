#!/bin/bash

CUDA_VISIBLE_DEVICES=1 ./run_all_V1.sh
CUDA_VISIBLE_DEVICES=1 ./run_all_V1_transfer.sh
CUDA_VISIBLE_DEVICES=1 ./run_all_V1_R.sh
CUDA_VISIBLE_DEVICES=1 ./run_all_V1_R_transfer.sh
