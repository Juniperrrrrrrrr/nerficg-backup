#!/bin/bash

source ~/miniconda3/etc/profile.d/conda.sh
conda activate nerficg

echo "🚀 [第二轮双卡并行] 开始训练 farm 和 gallery_pillar..."

# 显卡 0 跑 farm
echo "📅 GPU 0: farm 启动"
CUDA_VISIBLE_DEVICES=0 python ./scripts/train.py -c configs/ricoh360/farm.yaml > farm_train.log 2>&1 &

# 显卡 1 跑 gallery_pillar
echo "📅 GPU 1: gallery_pillar 启动"
CUDA_VISIBLE_DEVICES=1 python ./scripts/train.py -c configs/ricoh360/gallery_pillar.yaml > gallery_pillar_train.log 2>&1 &

echo "✅ 任务已提交。请使用 'tail -f farm_train.log' 查看进度。"
wait
echo "🎉 第二轮任务也全部跑完啦！"


