#!/bin/bash

LAMBDA_SPATIOTEMPORAL=(1 5 10)
LAMBDA_UNSUP_CYCLE=(1 5 10)

export CUDA_VISIBLE_DEVICES=1

cd ../..

for lambda_spatiotemporal in "${LAMBDA_SPATIOTEMPORAL[@]}"
do
    for lambda_unsup_cycle in "${LAMBDA_UNSUP_CYCLE[@]}"
    do
        python train.py --dataroot /mnt/kurnianto_hdd/dataset/Unsup_Recycle_GAN/dataset/viper_weather_HD/day2snow-256 \
        --model unsup_single --dataset_mode unaligned_scale --name d2snow_ws2_finetune_unsup_spatio${lambda_spatiotemporal}_unsupcycle${lambda_unsup_cycle}_size256-256_11-oct \
        --loadSizeW 256 --loadSizeH 256 --resize_mode rectangle --fineSizeW 256 --fineSizeH 256 \
        --crop_mode rectangle --which_model_netG resnet_6blocks --no_dropout --pool_size 0 \
        --lambda_spa_unsup_A $lambda_spatiotemporal --lambda_spa_unsup_B $lambda_spatiotemporal \
        --lambda_unsup_cycle_A $lambda_unsup_cycle --lambda_unsup_cycle_B $lambda_unsup_cycle \
        --lambda_cycle_A 10 --lambda_cycle_B 10 \
        --lambda_content_A 0 --lambda_content_B 0 \
        --batchSize 1 --noise_level 0.001  --niter_decay 0 --niter 20 \
        --use_tfboard
    done
done