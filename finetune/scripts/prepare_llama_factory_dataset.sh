python ./finetune/scripts/prepare_llama_factory_dataset.py \
    --raw-file ./data/training_set/data/llm/raw.jsonl \
    --image-root $IMAGE_ROOT

python ./finetune/scripts/prepare_llama_factory_dataset.py \
    --raw-file ./data/training_set/data/mllm/raw.jsonl \
    --image-root $IMAGE_ROOT
