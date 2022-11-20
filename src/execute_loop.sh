#!/bin/bash

COMMAND="-bb regnetx -bb regnet16x --single_img -ot 5 -ot 0 --double_img -op adam -op sgd --return_command --mem_avail 10000"
BASE_PYTHON="python main.py"

while true
do
    RES=$(python possible_fitted_mem_models.py ${COMMAND})
    if [ "$RES" = "stop"]; then
        break
    fi
    FULL_COMMAND="${BASE_PYTHON} ${RES}"
    exec ${FULL_COMMAND} &
    sleep 60
done

