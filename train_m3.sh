#!/bin/bash
nohup python src/fine_tune_vits.py \
	--early-stopping-epochs 5 \
	--model-name mbari-vars-dinov3-vitl16 \
	--base-model facebook/dinov3-vitl16-pretrain-lvd1689m \
	--add-rotations True \
	--remove-long-tail True \
	--num-epochs 15 \
	--min-images-per-class 100 \
	--raw-data /mnt/DeepSea-AI/data/VARS/crops/ \
	--filter-data /mnt/ML_SCRATCH/VARS_filtered_crops \
	--exclude-labels "marine snow" "marine organism"  > mbari-vars-vits-b16.log 2>&1 &
