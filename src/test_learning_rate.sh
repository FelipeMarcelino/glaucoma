epoch=100
models=("regnet" "mobile" "shuffle" "efficient" "regnet16" "regnet32" "inception")
optims=("adam" "sgd" "radam" "ranger")
lr=(0.01 0.001 0.0001 0.0005)
batch_size=16
early_start=100
output_tab=5

echo "One image"
for (( j=0; j<${#models[@]} ; j+=1 )) ; do
    echo "model: ${models[j]}"
    for (( i=0; i<${#optims[@]} ; i+=4 )) ; do
        for (( k=0; k<${#lr[@]} ; k+=1 )) ; do
            python main.py --optim ${optims[i]} --batch_size ${batch_size} \
            --backbone ${models[j]} --epochs ${epoch} --early_start ${early_start} --lr ${lr[k]} &
            python main.py --optim ${optims[i+1]} --batch_size ${batch_size} \
            --backbone ${models[j]} --epochs ${epoch} --early_start ${early_start} --lr ${lr[k]} &
            python main.py --optim ${optims[i+2]} --batch_size ${batch_size} \
            --backbone ${models[j]} --epochs ${epoch} --early_start ${early_start} --lr ${lr[k]} &
            python main.py --optim ${optims[i+3]} --batch_size ${batch_size} \
            --backbone ${models[j]} --epochs ${epoch} --early_start ${early_start} --lr ${lr[k]} &
        wait
        done
    done
done

# echo "Double image"
# for (( j=0; j<${#models[@]} ; j+=1 )) ; do
#     echo "model: ${models[j]}"
#     for (( i=0; i<${#optims[@]} ; i+=4 )) ; do
#         for (( k=0; k<${#lr[@]} ; k+=1 )) ; do
#             python main.py --optim ${optims[i]} --batch_size ${batch_size} \
#             --backbone ${models[j]} --epochs ${epoch} --early_start ${early_start} --double_img \
#             --lr ${lr[k]} &
#             python main.py --optim ${optims[i+1]} --batch_size ${batch_size} \
#             --backbone ${models[j]} --epochs ${epoch} --early_start ${early_start} --double_img \
#             --lr ${lr[k]} &
#             python main.py --optim ${optims[i+2]} --batch_size ${batch_size} \
#             --backbone ${models[j]} --epochs ${epoch} --early_start ${early_start} --double_img \
#             --lr ${lr[k]} &
#             python main.py --optim ${optims[i+3]} --batch_size ${batch_size} \
#             --backbone ${models[j]} --epochs ${epoch} --early_start ${early_start} --double_img \
#             --lr ${lr[k]} &
#         wait
#         done
#     done
# done

# echo "One image and tab"
# for (( j=0; j<${#models[@]} ; j+=1 )) ; do
#     echo "model: ${models[j]}"
#     for (( i=0; i<${#optims[@]} ; i+=4 )) ; do
#         for (( k=0; k<${#lr[@]} ; k+=1 )) ; do
#             python main.py --optim ${optims[i]} --batch_size ${batch_size} \
#             --backbone ${models[j]} --epochs ${epoch} --early_start ${early_start} \
#             --output_tab ${output_tab} --lr ${lr[k]} &
#             python main.py --optim ${optims[i+1]} --batch_size ${batch_size} \
#             --backbone ${models[j]} --epochs ${epoch} --early_start ${early_start} \
#             --output_tab ${output_tab} --lr ${lr[k]} &
#             python main.py --optim ${optims[i+2]} --batch_size ${batch_size} \
#             --backbone ${models[j]} --epochs ${epoch} --early_start ${early_start} \
#             --output_tab ${output_tab} --lr ${lr[k]} &
#             python main.py --optim ${optims[i+3]} --batch_size ${batch_size} \
#             --backbone ${models[j]} --epochs ${epoch} --early_start ${early_start} \
#             --output_tab ${output_tab} --lr ${lr[k]} &
#         wait
#         done
#     done
# done

# echo "Double image and tab"
# for (( j=0; j<${#models[@]} ; j+=1 )) ; do
#     echo "model: ${models[j]}"
#     for (( i=0; i<${#optims[@]} ; i+=4 )) ; do
#         for (( k=0; k<${#lr[@]} ; k+=1 )) ; do
#             python main.py --optim ${optims[i]} --batch_size ${batch_size} \
#             --backbone ${models[j]} --epochs ${epoch} --early_start ${early_start} --output_tab \
#                     ${output_tab} --double_img --lr ${lr[k]} &
#             python main.py --optim ${optims[i+1]} --batch_size ${batch_size} \
#             --backbone ${models[j]} --epochs ${epoch} --early_start ${early_start} --output_tab \
#                     ${output_tab} --double_img --lr ${lr[k]} &
#             python main.py --optim ${optims[i+2]} --batch_size ${batch_size} \
#             --backbone ${models[j]} --epochs ${epoch} --early_start ${early_start} --output_tab \
#                     ${output_tab} --double_img --lr ${lr[k]} &
#             python main.py --optim ${optims[i+3]} --batch_size ${batch_size} \
#             --backbone ${models[j]} --epochs ${epoch} --early_start ${early_start} --output_tab \
#                     ${output_tab} --double_img --lr ${lr[k]} &
#         wait
#         done
#     done
# done
