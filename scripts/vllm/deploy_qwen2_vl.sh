cuda=$1

CUDA_VISIBLE_DEVICES=${cuda} \
vllm serve ./models/Qwen2-VL-7B-Instruct/merged \
    --host 0.0.0.0 \
    --port 65533 \
    --trust-remote-code \
    --max-model-len 32768 \
    --limit-mm-per-prompt image=10 \
    --max-num-seqs 64 \
    --enforce-eager
