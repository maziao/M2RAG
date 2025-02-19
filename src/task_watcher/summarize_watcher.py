import os
import json
from tqdm import tqdm
from typing import Optional, Tuple, List
from src.utils import MultithreadManager
from src.task_watcher import TASK_WATCHER
from src.summarizer import build_summarizer
from src.task_watcher.watcher import TaskWatcher


@TASK_WATCHER.register_module
class SummarizeWatcher(TaskWatcher):
    def __init__(
        self,
        mongodb_url: Optional[str] = None,
        database_name: Optional[str] = None,
        batch_size: int = 0,
        summarizer_config: dict = {},
        caption_model_name: str = None,
        dry_run: bool = False,
        multi_thread_config: dict = {},
        log_dir: str = None,
        levels: List[int] = [1]
    ) -> None:
        super().__init__(mongodb_url, database_name, batch_size)        
        self.query_table = self.database['queries']
        self.webpage_table = self.database['webpages']
        self.image_table = self.database['images']
        self.aux_image_table = self.database['aux_images']
        self.summary_table = self.database['summaries']
        
        self.summarizer = build_summarizer(summarizer_config)
        
        self.caption_model_name = caption_model_name
        
        self.dry_run = dry_run
        
        self.manager = MultithreadManager(**multi_thread_config)
        
        assert log_dir is not None
        self.log_dir = os.path.join(log_dir, "summarize", self.summarizer.name)
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir, exist_ok=True)
        
        levels = list(set(levels))
        assert all([level <= 3 and level >= 1 for level in levels]) and len(levels) > 0
        self.levels = levels
        
        query_ids = []
        for level in self.levels:
            query_ids += [query['query_id'] for query in self.query_table.find({'level': level})]
        self.query_id_set = set(query_ids)
        
        self.post_init()
        
    def post_init(self):
        query_ids = []
        for level in self.levels:
            query_ids += [result['query_id'] for result in self.query_table.find({'level': level}, {'query_id': 1})]
        added_query_ids = [result['query_id'] for result in self.summary_table.find({'summarizer_id': self.summarizer.name}, {'query_id': 1})]
        new_query_ids = list(set(query_ids) - set(added_query_ids))
        if len(new_query_ids) > 0:
            self.summary_table.insert_many([
                {
                    'query_id': query_id,
                    'summarizer_id': self.summarizer.name,
                    'webpages': None,
                    'aux_images': None,
                    'summarize_label': False,
                    'model_input': None,
                    'model_response': None,
                    'usage': None,
                    'processed_response': None,
                    'placeholder_response': None,
                    'output_images': None,
                    'num_input_webpages': None,
                    'num_input_images': None,
                    'num_output_images': None,
                    'num_reference_webpages': None,
                    'evaluate_label': False,
                    'evaluate_result': None,
                    'evaluate_scores': None
                } for query_id in new_query_ids
            ])
            
    def _get_query_ids(self):
        uncompleted_query_ids = [summary['query_id'] for summary in self.summary_table.find({'summarizer_id': self.summarizer.name, 'summarize_label': False})]
        query_ids = list(self.query_id_set & set(uncompleted_query_ids))
        return query_ids
    
    def _get_query_ids_within_given_query_ids(self, query_ids):
        uncompleted_query_ids = []
        for query_id in query_ids:
            summary_sample = self.summary_table.find_one({'query_id': query_id, 'summarizer_id': self.summarizer.name}, {'summarize_label': 1})
            if summary_sample['summarize_label']:
                continue
            else:
                uncompleted_query_ids.append(query_id)
                if self.batch_size > 0 and len(uncompleted_query_ids) == self.batch_size:
                    break
        return uncompleted_query_ids
        
    def _watch_single_cycle(self, *args, **kwargs) -> Tuple[int, int]:
        # summarize for certain queries
        query_ids = kwargs.get('query_ids')
        if query_ids is not None:
            query_ids = self._get_query_ids_within_given_query_ids(query_ids=query_ids)
        else:
            query_ids = self._get_query_ids()
        
        # ordinary workflow
        for query_id in tqdm(query_ids, desc=f"Preparing knowledge bases for queries"):
            query = self.query_table.find_one({'query_id': query_id})
            if query['search_result'] is None or query['image_search_result'] is None:
                continue
            
            query_content = query['query_content']
            webpages = query['search_result']
            
            for webpage in webpages:
                webpage_id = webpage['webpage_id']
                webpage_result = self.webpage_table.find_one({'query_id': query_id, 'webpage_id': webpage_id})
                cleaned_webpage_splits = webpage_result['cleaned_webpage_splits']
                images = []
                for image in self.image_table.find({'query_id': query_id, 'webpage_id': webpage_id}):
                    if self.caption_model_name is None:
                        caption = image['caption']
                    elif 'all_captions' in image and self.caption_model_name in image['all_captions']:
                        caption = image['all_captions'][self.caption_model_name]
                    else:
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
            for image in self.aux_image_table.find({'query_id': query_id}):
                if self.caption_model_name is None:
                    caption = image['caption']
                elif 'all_captions' in image and image['all_captions'] is not None and self.caption_model_name in image['all_captions']:
                    caption = image['all_captions'][self.caption_model_name]
                else:
                    caption = None
                
                valid = caption is not None and image['final_score'] is not None
                aux_images.append({
                    'image_id': image['image_id'],
                    'image_url': image['image_url'],
                    'cached_image_url': image['cached_image_url'],
                    'valid': valid,
                    'final_score': image['final_score'],
                    'detailed_image_caption': caption
                })
            
            query_list = [{
                'query_id': query_id,
                'content': query_content,
                'webpages': webpages,
                'aux_images': aux_images
            }]
            
            self.manager.add_task(
                self.summarizer.summarize,
                os.path.join(self.log_dir, self.summarizer.name, f"{query_id}.json") if not self.dry_run else None,
                query_id,
                query_list=query_list,
                dry_run=self.dry_run
            )
        
        results = self.manager.execute_tasks()
        self.manager.clear_tasks()
        
        num_image_counter = {
            'webpage_images': {},
            'aux_images': {},
            'total_images': {}
        }
        num_webpage_counter = {}
        char_len_counter = 0
        
        for result in results:
            if result['result'] is not None and result['success']:
                if not self.dry_run:
                    self.summary_table.update_one(
                        filter={'query_id': result['id'], 'summarizer_id': self.summarizer.name},
                        update={
                            '$set': {
                                'summarize_label': True,
                                **result['result']
                            }
                        }
                    )
                else:
                    for key, value in result['result']['num_input_images'].items():
                        if value in num_image_counter[key]:
                            num_image_counter[key][value] += 1
                        else:
                            num_image_counter[key][value] = 1
                    
                    if result['result']['num_input_webpages']['filtered'] in num_webpage_counter:
                        num_webpage_counter[result['result']['num_input_webpages']['filtered']] += 1
                    else:
                        num_webpage_counter[result['result']['num_input_webpages']['filtered']] = 1
                    
                    char_len_counter += result['result']['num_input_webpages']['total_char_len']
                    print(json.dumps(result['result'], indent=4))
                    result['success'] = False
                    result['result'] = None
        
        if self.dry_run:
            webpage_images = {key: value for key, value in sorted(num_image_counter['webpage_images'].items(), key=lambda s: s[0])}
            aux_images = {key: value for key, value in sorted(num_image_counter['aux_images'].items(), key=lambda s: s[0])}
            total_images = {key: value for key, value in sorted(num_image_counter['total_images'].items(), key=lambda s: s[0])}
            webpages = {key: value for key, value in sorted(num_webpage_counter.items(), key=lambda s: s[0])}
            num_webpages = sum([key * value for key, value in webpages.items()])
            avg_webpage_char = char_len_counter // num_webpages
            print(f"webpage images: {json.dumps(webpage_images, indent=4)}")
            print(f"aux images: {json.dumps(aux_images, indent=4)}")
            print(f"total images: {json.dumps(total_images, indent=4)}")
            print(f"webpages: {json.dumps(webpages, indent=4)}")
            print(f"avg char len: {avg_webpage_char}")

        return sum([int(result['result'] is not None and result['success']) for result in results]), len(results)
