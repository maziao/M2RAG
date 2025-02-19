import os
import time
import yaml
import logging
import argparse
from src.agent import M2RAGAgent
from src.utils import configure_log


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config-file', type=str, default='./src/config/gpt_4o.yaml')
    parser.add_argument('--cache-dir', type=str, default='./log/demo/cache')
    parser.add_argument('--image-dir', type=str, default='images')
    parser.add_argument('--query', type=str, required=True)
    parser.add_argument('--session-id', type=str, default=None)
    args = parser.parse_args()
    
    with open(args.config_file) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        
    if args.session_id is not None:
        session_id = args.session_id
    else:
        session_id = time.strftime('%Y%m%d-%H%M%S')
        
    configure_log(name=f"demo/{session_id}")
    
    logging.info(f"[!] session_id: {session_id} [!]")
    cache_dir = os.path.join(args.cache_dir, session_id)
    image_dir = os.path.join(args.image_dir, session_id)
    
    config['cache_dir'] = cache_dir
    config['image_dir'] = image_dir
    
    agent = M2RAGAgent(**config)
    
    topics = [{'query_id': 0, 'content': args.query}]
    result = agent.search_pipeline(topics=topics, session_id=session_id)
    
    print(result)
    