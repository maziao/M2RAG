import os
import json
import yaml
import argparse
from src.utils import MultithreadManager
from src.summarizer import build_summarizer


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-file', type=str, default='./data/dev_set/dev_set.jsonl')
    parser.add_argument('--config-file', type=str, default='./src/config/summarize_custom_mllm.yaml')
    args = parser.parse_args()
    
    # load dataset
    with open(args.data_file, 'r+', encoding='utf-8') as f:
        dataset = [json.loads(line) for line in f.readlines()]
    
    # load config
    with open(args.config_file) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    
    summarizer = build_summarizer(config['summarizer_config'])
    
    manager = MultithreadManager(**config['multi_thread_config'])
    
    # configure log_dir
    log_dir = config['log_dir']
    assert log_dir is not None
    log_dir = os.path.join(log_dir, "summarize", summarizer.name)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)
    
    # select samples
    levels = list(set(config['levels']))
    assert all([level <= 3 and level >= 1 for level in levels]) and len(levels) > 0
    selected_samples = [sample for sample in dataset if sample['level'] in levels]
    
    for sample in selected_samples:
        manager.add_task(
            summarizer.summarize,
            os.path.join(log_dir, f"{sample['query_id']}.json"),
            sample['query_id'],
            query_list=[sample],
            dry_run=False
        )
    
    results = manager.execute_tasks()
    manager.clear_tasks()
    
    overall = len(results)
    success = sum([result['success'] and result['result'] is not None for result in results])
    print(f"[!] Overall: {overall}; Success: {success}; Success Rate: {round(100 * success / overall)}%")
