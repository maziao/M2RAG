config=$1
num_gpu=$2
cuda=$3

FORCE_TORCHRUN=1 \
NPROC_PER_NODE=${num_gpu} \
CUDA_VISIBLE_DEVICES=${cuda} llamafactory-cli train ${config}