epoch=100
models=("regnet" "mobile" "shuffle" "efficient" "shuffle" "regnet16" "regnet32" "vit" "inception")
optims=("adam" "sgd" "radam" "ranger")
lr=(0.01 0.001 0.0001 0.0005)
batch_size=16
early_start=100
output_tab=5

for (( j=0; j<${#models[@]} ; j+=1 )) ; do
    for (( i=0; i<${#optims[@]} ; i+=2 )) ; do
        for (( k=0; k<${#lr[@]} ; k+=1 )) ; do
            echo "${models[j]} + ${optims[i]} + ${optims[i+1]} + ${lr[k]}"
            python main.py --optim ${optims[i]} --batch_size ${batch_size} \
            --model_name ${models[j]} --epochs ${epoch} --early_start ${early_start} --lr ${lr} &
            python main.py --optim ${optims[i+1]} --batch_size ${batch_size} \
            --model_name ${models[j]} --epochs ${epoch} --early_start ${early_start} --lr ${lr} &

        wait
        done
    done
done


for (( j=0; j<${#models[@]} ; j+=1 )) ; do
    for (( i=0; i<${#optims[@]} ; i+=2 )) ; do
        for (( k=0; k<${#lr[@]} ; k+=1 )) ; do
            echo "${models[j]} + ${optims[i]} + ${optims[i+1]} + ${lr[k]}"
            python main.py --optim ${optims[i]} --batch_size ${batch_size} \
            --model_name ${models[j]} --epochs ${epoch} --early_start ${early_start} --double_img \
            --lr ${lr} &
            python main.py --optim ${optims[i+1]} --batch_size ${batch_size} \
            --model_name ${models[j]} --epochs ${epoch} --early_start ${early_start} --double_img \
            --lr ${lr} &
        wait
        done
    done
done

for (( j=0; j<${#models[@]} ; j+=1 )) ; do
    for (( i=0; i<${#optims[@]} ; i+=2 )) ; do
        for (( k=0; k<${#lr[@]} ; k+=1 )) ; do
            echo "${models[j]} + ${optims[i]} + ${optims[i+1]} + ${lr[k]}"
            python main.py --optim ${optims[i]} --batch_size ${batch_size} \
            --model_name ${models[j]} --epochs ${epoch} --early_start ${early_start} \
            --output_tab ${output_tab} --lr ${lr} &
            python main.py --optim ${optims[i+1]} --batch_size ${batch_size} \
            --model_name ${models[j]} --epochs ${epoch} --early_start ${early_start} \
            --output_tab ${output_tab} --lr ${lr} &
        wait
        done
    done
done



for (( j=0; j<${#models[@]} ; j+=1 )) ; do
    for (( i=0; i<${#optims[@]} ; i+=2 )) ; do
        for (( k=0; k<${#lr[@]} ; k+=1 )) ; do
            echo "${models[j]} + ${optims[i]} + ${optims[i+1]} + ${lr[k]}"
            python main.py --optim ${optims[i]} --batch_size ${batch_size} \
            --model_name ${models[j]} --epochs ${epoch} --early_start ${early_start} --output_tab \
                    ${output_tab} --double_img --lr ${lr} &
            python main.py --optim ${optims[i+1]} --batch_size ${batch_size} \
            --model_name ${models[j]} --epochs ${epoch} --early_start ${early_start} --output_tab \
                    ${output_tab} --double_img --lr ${lr} &
        wait
        done
    done
done
