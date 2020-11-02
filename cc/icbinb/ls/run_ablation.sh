#!/bin/bash
seeds=( 0 1 2 3 4 5 6 7 8 9 )
losses=( XE LS CC RLS RCC )
losses=( XE LS CC )
wds=( 0.0 0.0001 )
dps=( Yes No )
bns=( Yes No )
for seed in "${seeds[@]}"; do
for loss in "${losses[@]}"; do
for dp in "${dps[@]}"; do
for bn in "${bns[@]}"; do
for wd in "${wds[@]}"; do
    sbatch gpu_run.sh "cifar10_model.py --loss=$loss --dp=$dp --bn=$bn --seed=$seed --wd=$wd --save=Yes"
done
done
done
done
done
