#!/bin/bash

COMMAND="-bb regnetx -bb regnet16x --single_img -ot 5 -ot 0 --double_img -op adam -op sgd --return_command"
BASE_PYTHON="python main.py"
BASE_MEM="--mem_avail"
GPU_ID=0

while true
do

    mem_avail=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i ${GPU_ID})
    RES=$(python possible_fitted_mem_models.py ${COMMAND} ${BASE_MEM} ${mem_avail})
    if [[ "$RES" = "Stop" ]]; then
        break
    fi
    if [[ "$RES" != "Full" ]]; then
        FULL_COMMAND="${BASE_PYTHON} ${RES}"
        nohup ${FULL_COMMAND} &
    fi
    sleep 60
done


echo "Done!"
