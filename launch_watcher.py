import json
import yaml
import argparse
from src.task_watcher import build_task_watcher


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config-file', type=str, required=True)
    parser.add_argument('--kwargs-file', type=str, default=None)
    parser.add_argument('--terminate-on-cycle-end', action='store_true')
    args = parser.parse_args()
    
    with open(args.config_file) as f:
        watcher_config = yaml.load(f, Loader=yaml.FullLoader)
        
    if args.kwargs_file is not None:
        assert args.kwargs_file.endswith('.yaml') or args.kwargs_file.endswith('.json')
        if args.kwargs_file.endswith('.yaml'):
            with open(args.kwargs_file) as f:
                kwargs = yaml.load(f, Loader=yaml.FullLoader)
        else:
            with open(args.kwargs_file, 'r+', encoding='utf-8') as f:
                kwargs = json.load(f)
    else:
        kwargs = {}
        
    watcher = build_task_watcher(watcher_config)
    
    watcher.watch_forever(terminate_on_cycle_end=args.terminate_on_cycle_end, **kwargs)
    