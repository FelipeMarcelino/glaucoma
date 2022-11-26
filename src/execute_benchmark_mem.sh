#!/bin/bash
epoch=2
models=("regnet" "mobile" "shuffle" "efficient" "regnet16" "regnet32" "inception" "regnetx" "regnet16x" "regnet32x" "vit" "resnet")
optims=("adam")
lr=(0.001)
batch_size=16
early_start=2
output_tab=5


for (( j=0; j<${#models[@]} ; j+=1 )) ; do
    for (( i=0; i<${#optims[@]} ; i+=1 )) ; do
        for (( k=0; k<${#lr[@]} ; k+=1 )) ; do
            python main.py --debug --optim ${optims[i]} --batch_size ${batch_size} \
            --backbone ${models[j]} --epochs ${epoch} --early_start ${early_start} --lr ${lr[k]} &
            wait
            python main.py --debug --optim ${optims[i]} --batch_size ${batch_size} \
            --backbone ${models[j]} --epochs ${epoch} --early_start ${early_start} --double_img \
            --lr ${lr[k]} &
            wait
            python main.py --debug --optim ${optims[i]} --batch_size ${batch_size} \
            --backbone ${models[j]} --epochs ${epoch} --early_start ${early_start} \
            --output_tab ${output_tab} --lr ${lr[k]} &
            wait
            python main.py debug --optim ${optims[i]} --batch_size ${batch_size} \
            --backbone ${models[j]} --epochs ${epoch} --early_start ${early_start} --output_tab \
                    ${output_tab} --double_img --lr ${lr[k]} &
        wait
        done
    done
done

