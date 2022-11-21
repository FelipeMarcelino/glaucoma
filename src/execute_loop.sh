#!/bin/bash
BACKBONE="-bb"
OPTIM="-op"
LR="--lr"
BATCH_SIZE="-bs"
FRAC_VAL="-fv"
K_FOLD="--k_fold"
EARLY_START="-es"
EPOCHS="-ep"
OUTPUT_TAB="-ot"
BASE_PYTHON="python main.py"
BASE_MEM="--mem_avail"

LIST_BACKBONES=("regnetx" "regnet16x" "regnet32x" "regnet" "regnet16" "regnet32" "vit" "inception" "resnet" "shuffle" "mobile" "efficient")
LIST_OPTIM=("adam" "ranger" "sgd" "ranger")
LIST_ARQ=("double_img" "single_img") # single_img or double_img
LIST_BATCH_SIZE=(16)
LIST_FRAC_VAL=(0.2)
LIST_OUTPUT_TAB=(0 5)
LIST_EPOCHS=(100)
LIST_EARLY_STOP=(100)
LIST_LR=(0.01 0.001 0.0001 0.0005)

BASE_COMMAND="--return_command "

for (( i=0; i<${#LIST_BACKBONES[@]} ; i+=1 )) ; do
    BASE_COMMAND+="${BACKBONE} "
    BASE_COMMAND+="${LIST_BACKBONES[i]} "
done

for (( i=0; i<${#LIST_OPTIM[@]} ; i+=1 )) ; do
    BASE_COMMAND+="${OPTIM} "
    BASE_COMMAND+="${LIST_OPTIM[i]} "
done

for (( i=0; i<${#LIST_BATCH_SIZE[@]} ; i+=1 )) ; do
    BASE_COMMAND+="${BATCH_SIZE} "
    BASE_COMMAND+="${LIST_BATCH_SIZE[i]} "
done

for (( i=0; i<${#LIST_EARLY_STOP[@]} ; i+=1 )) ; do
    BASE_COMMAND+="${EARLY_START} "
    BASE_COMMAND+="${LIST_EARLY_STOP[i]} "
done


for (( i=0; i<${#LIST_EPOCHS[@]} ; i+=1 )) ; do
    BASE_COMMAND+="${EPOCHS} "
    BASE_COMMAND+="${LIST_EPOCHS[i]} "
done

for (( i=0; i<${#LIST_FRAC_VAL[@]} ; i+=1 )) ; do
    BASE_COMMAND+="${FRAC_VAL} "
    BASE_COMMAND+="${LIST_FRAC_VAL[i]} "
done

for (( i=0; i<${#LIST_OUTPUT_TAB[@]} ; i+=1 )) ; do
    BASE_COMMAND+="${OUTPUT_TAB} "
    BASE_COMMAND+="${LIST_OUTPUT_TAB[i]} "
done

for (( i=0; i<${#LIST_LR[@]} ; i+=1 )) ; do
    BASE_COMMAND+="${LR} "
    BASE_COMMAND+="${LIST_LR[i]} "
done

for (( i=0; i<${#LIST_ARQ[@]} ; i+=1 )) ; do
    BASE_COMMAND+="--"
    BASE_COMMAND+="${LIST_ARQ[i]} "
done

BASE_PYTHON="python main.py"
BASE_MEM="--mem_avail"
GPU_ID=0

while true
do
    mem_avail=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i ${GPU_ID})
    RES=$(python possible_fitted_mem_models.py ${BASE_COMMAND} ${BASE_MEM} ${mem_avail})
    if [[ "$RES" = "Stop" ]]; then
        break
    fi
    if [[ "$RES" != "Full" ]]; then
        FULL_COMMAND="${BASE_PYTHON} ${RES}"
        echo ${FULL_COMMAND}
        nohup ${FULL_COMMAND} &
    fi
    sleep 60
done


echo "Done!"
