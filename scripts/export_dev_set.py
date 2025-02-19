import os
import json
import pymongo
import argparse
from tqdm import tqdm


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--mongodb-url', type=str, default='mongodb://0.0.0.0:27017')
    parser.add_argument('--database-name', type=str, default='dev_set')
    parser.add_argument('--output-file', type=str, default='./data/dev_set/dev_set-custom.jsonl')
    parser.add_argument('--levels', type=int, nargs='*', default=[1, 2, 3])
    args = parser.parse_args()
    
    output_dir = os.path.dirname(args.output_file)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    # connect with database
    mongodb_client = pymongo.MongoClient(args.mongodb_url)
    database = mongodb_client["m2rag"][args.database_name]
    
    query_table = database['queries']
    webpage_table = database['webpages']
    image_table = database['images']
    aux_image_table = database['aux_images']
    summary_table = database['summaries']
    
    levels = list(set(args.levels))
    assert all([level <= 3 and level >= 1 for level in levels]) and len(levels) > 0
    
    query_ids = []
    for level in levels:
        query_ids += [result['query_id'] for result in query_table.find({'level': level}, {'query_id': 1})]
    
    dev_set = []
    for query_id in tqdm(query_ids, desc=f"Preparing knowledge bases for queries"):
        query = query_table.find_one({'query_id': query_id})
        if query['search_result'] is None or query['image_search_result'] is None:
            continue
        
        query_content = query['query_content']
        webpages = query['search_result']
        
        for webpage in webpages:
            webpage_id = webpage['webpage_id']
            webpage_result = webpage_table.find_one({'query_id': query_id, 'webpage_id': webpage_id})
            cleaned_webpage_splits = webpage_result['cleaned_webpage_splits']
            images = []
            for image in image_table.find({'query_id': query_id, 'webpage_id': webpage_id}):
                caption = image['caption']
                
                valid = caption is not None and image['split_id'] is not None and image['final_score'] is not None
                images.append({
                    'split_id': image['split_id'],
                    'image_id': image['image_id'],
                    'image_url': image['image_url'],
                    'valid': valid,
                    'cached_image_url': image['cached_image_url'],
                    'final_score': image['final_score'],
                    'detailed_image_caption': caption
                })
            images = sorted(images, key=lambda s: s['image_id'], reverse=False)
            
            webpage.update({
                'cleaned_webpage_splits': cleaned_webpage_splits,
                'images': images
            })
            
        aux_images = []
        for image in aux_image_table.find({'query_id': query_id}):
            caption = image['caption']
            
            valid = caption is not None and image['final_score'] is not None
            aux_images.append({
                'image_id': image['image_id'],
                'image_url': image['image_url'],
                'cached_image_url': image['cached_image_url'],
                'valid': valid,
                'final_score': image['final_score'],
                'detailed_image_caption': caption
            })
        
        sample = {
            'query_id': query_id,
            'content': query_content,
            'webpages': webpages,
            'aux_images': aux_images,
            'level': query['level']
        }
        
        dev_set.append(sample)
        
    with open(args.output_file, 'w+', encoding='utf-8') as f:
        for sample in dev_set:
            f.write(json.dumps(sample, ensure_ascii=False) + '\n')
