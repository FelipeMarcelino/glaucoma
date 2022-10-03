epoch=100
models=("regnet" "mobile" "shuffle" "efficient" "shuffle" "regnet16" "regnet32" "vit" "inception")
optims=("adam" "sgd" "radam" "ranger")
batch_size=16
early_start=100
output_tab=5

for (( j=0; j<${#models[@]} ; j+=1 )) ; do
    for (( i=0; i<${#optims[@]} ; i+=2 )) ; do
        echo "${models[j]} + ${optims[i]} + ${optims[i+1]}"
        python main.py --optim ${optims[i]} --batch_size ${batch_size} \
        --model_name ${models[j]} --epochs ${epoch} --early_start ${early_start} &

        wait
    done
done


for (( j=0; j<${#models[@]} ; j+=1 )) ; do
    for (( i=0; i<${#optims[@]} ; i+=2 )) ; do
        echo "${models[j]} + ${optims[i]} + ${optims[i+1]}"
        python main.py --optim ${optims[i]} --batch_size ${batch_size} \
        --model_name ${models[j]} --epochs ${epoch} --early_start ${early_start} --double_img &

        wait
    done
done

for (( j=0; j<${#models[@]} ; j+=1 )) ; do
    for (( i=0; i<${#optims[@]} ; i+=2 )) ; do
        echo "${models[j]} + ${optims[i]} + ${optims[i+1]}"
        python main.py --optim ${optims[i]} --batch_size ${batch_size} \
        --model_name ${models[j]} --epochs ${epoch} --early_start ${early_start} --output_tab ${output_tab} &

        wait
    done
done



for (( j=0; j<${#models[@]} ; j+=1 )) ; do
    for (( i=0; i<${#optims[@]} ; i+=2 )) ; do
        echo "${models[j]} + ${optims[i]} + ${optims[i+1]}"
        python main.py --optim ${optims[i]} --batch_size ${batch_size} \
        --model_name ${models[j]} --epochs ${epoch} --early_start ${early_start} --output_tab \
                ${output_tab} --double_img &

        wait
    done
done


