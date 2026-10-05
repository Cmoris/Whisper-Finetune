#!/usr/bin/env bash
set -e

cd "$(dirname "$0")/.."

# large-v2 教师 -> small 学生；删除 --student_model 参数可使用原始 Transformer。
python train_distil.py \
    --teacher_model openai/whisper-large-v2 \
    --student_model openai/whisper-small \
    --train_data dataset/train.json \
    --test_data dataset/test.json \
    --output_dir output/distillation \
    --language Chinese \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --learning_rate 1e-4 \
    --num_train_epochs 3 \
    --lambda 0.5 \
    --temperature 1.5 \
    "$@"