#!/bin/bash

# 1. 激活 Conda 环境 (确保路径指向你的 miniconda)
source ~/miniconda3/etc/profile.d/conda.sh
conda activate nerficg

echo "🚀 [双卡模式] 开始并行训练 Ricoh360 场景..."

# 2. 在显卡 0 上运行 cat_tower 场景 (加上 & 表示后台运行)
echo "📅 显卡 0 正在启动: cat_tower"
CUDA_VISIBLE_DEVICES=0 python ./scripts/train.py -c configs/ricoh360/cat_tower.yaml > cat_tower_train.log 2>&1 &

# 3. 在显卡 1 上运行 center 场景
echo "📅 显卡 1 正在启动: center"
CUDA_VISIBLE_DEVICES=1 python ./scripts/train.py -c configs/ricoh360/center.yaml > center_train.log 2>&1 &

echo "-------------------------------------------------------"
echo "✅ 两个任务已提交后台！"
echo "📈 你可以输入 'tail -f cat_tower_train.log' 查看第一个任务进度。"
echo "📈 你可以输入 'tail -f center_train.log' 查看第二个任务进度。"
echo "-------------------------------------------------------"

# 等待所有后台任务完成
wait
echo "🎉 所有训练任务已顺利完成！"


