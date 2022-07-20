# epoch=100
# models=("regnet" "mobile" "shuffle" "efficient" "shuffle")
# optims=("adam" "sgd" "radam" "ranger")
# k_fold=5
# batch_size=32

# for (( j=0; j<${#models[@]} ; j+=1 )) ; do
#     for (( i=0; i<${#optims[@]} ; i+=2 )) ; do
#         echo "${models[j]} + ${optims[i]} + ${optims[i+1]}"
#         python main.py --optim ${optims[i]} --batch_size ${batch_size} --k_fold ${k_fold}\
#         --model_name ${models[j]} &
#         python main.py --optim ${optims[i+1]} --batch_size ${batch_size} --k_fold ${k_fold}\
#         --model_name ${models[j]} &

#         wait
#     done
# done

# epoch=100
# models=("regnet" "mobile" "shuffle" "efficient" "shuffle")
# optims=("adam" "sgd" "radam" "ranger")
# k_fold=5
# batch_size=32
# lr=0.001

# for (( j=0; j<${#models[@]} ; j+=1 )) ; do
#     for (( i=0; i<${#optims[@]} ; i+=2 )) ; do
#         echo "${models[j]} + ${optims[i]} + ${optims[i+1]}"
#         python main.py --optim ${optims[i]} --batch_size ${batch_size} --k_fold ${k_fold}\
#         --model_name ${models[j]} --lr ${lr} &
#         python main.py --optim ${optims[i+1]} --batch_size ${batch_size} --k_fold ${k_fold}\
#         --model_name ${models[j]} --lr ${lr} &

#         wait
#     done
# done


epoch=100
models=("regnet")
optims=("adam" "sgd" "radam" "ranger")
k_fold=5
batch_size=32
lr=0.001

for (( j=0; j<${#models[@]} ; j+=1 )) ; do
    for (( i=0; i<${#optims[@]} ; i+=2 )) ; do
        echo "${models[j]} + ${optims[i]} + ${optims[i+1]}"
        python main.py --optim ${optims[i]} --batch_size ${batch_size} --k_fold ${k_fold}\
        --model_name ${models[j]} --lr ${lr} --multi_input &
        python main.py --optim ${optims[i+1]} --batch_size ${batch_size} --k_fold ${k_fold}\
        --model_name ${models[j]} --lr ${lr} --multi_input &

        wait
    done
done

for (( j=0; j<${#models[@]} ; j+=1 )) ; do
    for (( i=0; i<${#optims[@]} ; i+=1 )) ; do
        echo "${models[j]} + ${optims[i]} + ${optims[i+1]}"
        python main.py --optim ${optims[i]} --batch_size ${batch_size} --k_fold ${k_fold}\
        --model_name ${models[j]} --lr ${lr} --multi_input --double_img &
        wait
    done
done

