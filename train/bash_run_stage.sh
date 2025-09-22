#!/bin/bash

# -------------------
# 配置参数 - 直接在脚本中设置固定值
# -------------------
# GPU ID参数 - 直接设置为0，表示使用第一个GPU
GPU=0
# 分割索引参数 - 直接设置为0
SPLIT_IDX=0

# -------------------
# 默认参数配置
# -------------------
# 数据集相关
DATASET="food101"
FEWSHOT_SEED="seed0"
N_SHOT=16
NUM_TRAIN_EPOCH=200
target_class_idx=0

# -------------------
# 路径配置 (原local.yaml内容)
# -------------------
# 数据目录
FEWSHOT_DATA_DIR="/AutoPrompt++/data/sheep/real_train_fewshot/seed0"
# 预训练模型路径
PRETRAINED_MODEL_PATH="/hy-tmp/sd21/models--stabilityai--stable-diffusion-2-1/snapshots/5cae40e6a2745ae2b01ad92ae5043f95f23644d6"
# 标题数据路径
CAPTION_PATH="/AutoPrompt++/AutoPrompt/captions/cheese_plate_captions.json"

### ------------------
### 第一阶段: 数据集级训练
### ------------------
echo "开始数据集级训练..."
echo "使用GPU: $GPU"
echo "使用标题数据路径: $CAPTION_PATH"
echo "使用数据目录: $FEWSHOT_DATA_DIR"
echo "使用预训练模型: $PRETRAINED_MODEL_PATH"

CUDA_VISIBLE_DEVICES=$GPU accelerate launch autoprompt.py \
    --dataset=$DATASET \
    --fewshot_seed=$FEWSHOT_SEED \
    --train_batch_size=8 \
    --gradient_accumulation_steps=1 \
    --learning_rate=1e-4 \
    --lr_scheduler="cosine" \
    --lr_warmup_steps=100 \
    --num_train_epochs=$NUM_TRAIN_EPOCH \
    --report_to="tensorboard" \
    --train_text_encoder=True \
    --is_tqdm=True \
    --output_dir=outputs \
    --n_shot=$N_SHOT \
    --target_class_idx=$target_class_idx \
    --resume_from_checkpoint=None \
    --caption_path=$CAPTION_PATH \
    --fewshot_data_dir=$FEWSHOT_DATA_DIR \
    --pretrained_model_name_or_path=$PRETRAINED_MODEL_PATH

if [ $? -ne 0 ]; then
    echo "错误: 数据集级训练失败"
    exit 1
fi

### ------------------
### 第二阶段: 类别级训练
### ------------------
# 获取类别名称
CLASS_NAME=$(python3 -c "from util_data import SUBSET_NAMES; print(SUBSET_NAMES['${DATASET}'][$target_class_idx])")
CLASS_DIR="${FEWSHOT_DATA_DIR}/${CLASS_NAME}"

# 检查类别文件夹是否存在且非空
if [ -d "$CLASS_DIR" ] && [ "$(ls -A $CLASS_DIR)" ]; then
    echo "开始类别级训练，类别: $CLASS_NAME"
    echo "使用GPU: $GPU"
    echo "使用标题数据路径: $CAPTION_PATH"
    echo "使用数据目录: $FEWSHOT_DATA_DIR"
    echo "使用预训练模型: $PRETRAINED_MODEL_PATH"
    
    CUDA_VISIBLE_DEVICES=$GPU accelerate launch train.py \
        --dataset=$DATASET \
        --fewshot_seed=$FEWSHOT_SEED \
        --train_batch_size=8 \
        --gradient_accumulation_steps=1 \
        --learning_rate=1e-4 \
        --lr_scheduler="cosine" \
        --lr_warmup_steps=100 \
        --num_train_epochs=$NUM_TRAIN_EPOCH \
        --report_to="tensorboard" \
        --train_text_encoder=True \
        --is_tqdm=True \
        --output_dir=outputs \
        --n_shot=$N_SHOT \
        --target_class_idx=$target_class_idx \
        --resume_from_checkpoint=None \
        --caption_path=$CAPTION_PATH \
        --fewshot_data_dir=$FEWSHOT_DATA_DIR \
        --pretrained_model_name_or_path=$PRETRAINED_MODEL_PATH
        
    if [ $? -ne 0 ]; then
        echo "错误: 类别级训练失败"
        exit 1
    fi
else
    echo "错误: 未找到类别 $CLASS_NAME 的数据"
    exit 1
fi

echo "数据集级和类别级训练均已成功完成"