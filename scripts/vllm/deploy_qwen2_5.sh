cuda=$1

CUDA_VISIBLE_DEVICES=${cuda} \
vllm serve ./models/Qwen2.5-7B-Instruct/merged \
    --host 0.0.0.0 \
    --port 65534 \
    --trust-remote-code \
    --max-model-len 32768 \
    --max-num-seqs 64 \
    --enforce-eager
