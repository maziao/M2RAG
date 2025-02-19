cuda=$1

CUDA_VISIBLE_DEVICES=${cuda} \
vllm serve ./models/Llama-3.1-8B-Instruct/merged \
    --host 0.0.0.0 \
    --port 65534 \
    --trust-remote-code \
    --max-model-len 65536 \
    --max-num-seqs 64 \
    --enforce-eager
