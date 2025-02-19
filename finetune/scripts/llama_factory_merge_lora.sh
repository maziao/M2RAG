config=$1
cuda=$2

CUDA_VISIBLE_DEVICES=${cuda} llamafactory-cli export ${config}