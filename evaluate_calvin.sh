#!/bin/bash

export MESA_GL_VERSION_OVERRIDE=3.3

python3 evaluate_calvin.py \
    --eval_dir ${EVAL_DIR} \
    --mae_ckpt_path ${MAE_CKPT_PATH} \
    --policy_ckpt_path ${POLICY_CKPT_PATH} \
    --configs_path logs/configs.json \
    --dataset_dir task_ABCD_D/ \
    ${@:1}