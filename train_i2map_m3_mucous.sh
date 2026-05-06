#!/bin/bash

python src/fine_tune_vits.py \
	--early-stopping-epochs 2 \
	--model-name mbari-i2mapm3-dino-vitb8 \
	--base-model facebook/dino-vitb8 \
	--num-epochs 20 --add-rotations True \
        --raw-data \
	/mnt/DeepSea-AI/data/VARS/training-data/crops \
	/mnt/DeepSea-AI/data/i2MAP/training-data/i2map/crops/ \
	/mnt/DeepSea-AI/data/i2MAP/training-data/i2mapbulk/mbari-i2map-vits-b8-20251008-vss/crops \
	/mnt/DeepSea-AI/data/i2MAP/training-data/i2mapbulk/crops/ \
        --filter-data /mnt/ML_SCRATCH/i2mapm3_filtered_crops \
	--remap train_class_remap.json \
	--exclude-labels "sinkerunverified" "marine snow" "marine organism" "cra" "Unknown" "ghost tail" "object" "mucous web" "tentacle" "DeepPIV" "ROV" "Actiniaria" "DeepPIV 2.0" "DeepPIV 3.0" "Midwater Respirometry System"  "Midwater Respirometry System"
