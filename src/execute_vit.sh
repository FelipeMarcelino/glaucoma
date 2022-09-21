epoch=100
models=("vit")
optims=("adam" "sgd" "radam" "ranger")
k_fold=5
batch_size=32

for (( j=0; j<${#models[@]} ; j+=1 )) ; do
    for (( i=0; i<${#optims[@]} ; i+=2 )) ; do
        echo "${models[j]} + ${optims[i]} + ${optims[i+1]}"
        python main.py --optim ${optims[i]} --batch_size ${batch_size} --k_fold ${k_fold}\
        --model_name ${models[j]} --epochs ${epoch} &
        python main.py --optim ${optims[i+1]} --batch_size ${batch_size} --k_fold ${k_fold}\
        --model_name ${models[j]} --epochs ${epoch} &

        wait
    done
done

epoch=100
models=("vit")
optims=("adam" "sgd" "radam" "ranger")
k_fold=5
batch_size=32
lr=0.001

for (( j=0; j<${#models[@]} ; j+=1 )) ; do
    for (( i=0; i<${#optims[@]} ; i+=2 )) ; do
        echo "${models[j]} + ${optims[i]} + ${optims[i+1]}"
        python main.py --optim ${optims[i]} --batch_size ${batch_size} --k_fold ${k_fold}\
        --model_name ${models[j]} --lr ${lr} --epochs ${epoch} &
        python main.py --optim ${optims[i+1]} --batch_size ${batch_size} --k_fold ${k_fold}\
        --model_name ${models[j]} --lr ${lr} --epochs ${epoch} &

        wait
    done
done
