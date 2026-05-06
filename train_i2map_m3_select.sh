#!/bin/bash

python src/fine_tune_vits.py \
	--early-stopping-epochs 3 \
	--model-name mbari-i2mapm3-vits-b8 \
	--base-model facebook/dino-vitb8 \
	--num-epochs 50 --add-rotations True \
        --raw-data \
	/mnt/DeepSea-AI/data/M3/crops/ \
	/mnt/ML_SCRATCH/i2map/Baseline/crops \
        /mnt/ML_SCRATCH/i2mapbulk/crops/ \
        --filter-data /mnt/ML_SCRATCH/M3_filtered_crops \
	--remap train_class_remap.json \
	--exclude-labels "marine snow" "marine organism" "cra" "Unknown" "ghost tail" "object" "mucous web" "tentacle" "DeepPIV" "ROV" "Actiniaria" "DeepPIV 2.0" "equipment" "DeepPIV 3.0" "Apolemia body" "Midwater Respirometry System" "Detritus Sampler" "Bathochordaeus mcnutti inner filter" "Bathochordaeus mcnutti outer filter" "Bathochordaeus stygius outer filter" "Bathochordaeus stygius inner filter" "Bathochordaeus inner filter" "Bathochordaeus outer filter" "Suction Sampler" "Corymorpha" "Midwater Respirometry System"
