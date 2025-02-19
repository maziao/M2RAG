import os
import re
from typing import Optional, Tuple, List
from src.task_watcher import TASK_WATCHER
from src.task_watcher.watcher import TaskWatcher
from src.image_handler.image_caption import build_image_caption
from src.utils import MultithreadManager


IMAGE_PLACEHOLDER = "<IMAGE_PLACEHOLDER>"


@TASK_WATCHER.register_module
class CaptionTaskWatcher(TaskWatcher):
    def __init__(
        self,
        mongodb_url: Optional[str] = None,
        database_name: Optional[str] = None,
        batch_size: int = 0,
        caption_config: dict = {},
        caption_model_name: str = None,
        use_orig_image: bool = False,
        caption_aux_images: bool = False,
        log_dir: str = None,
        multi_thread_config: dict = {}
    ) -> None:
        super().__init__(mongodb_url, database_name, batch_size)
        self.query_table = self.database['queries']
        self.webpage_table = self.database['webpages']
        self.image_table = self.database['images']
        self.aux_image_table = self.database['aux_images']
        self.caption_agent = build_image_caption(caption_config)
        self.caption_model_name = caption_model_name
        self.use_orig_image = use_orig_image
        self.caption_aux_images = caption_aux_images
        self.manager = MultithreadManager(**multi_thread_config)
        
        assert log_dir is not None
        self.log_dir = os.path.join(log_dir, "caption_image")
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir, exist_ok=True)
    
    @staticmethod
    def _remove_image_placeholders(text: str):
        image_indices = re.findall(r"<IMAGE_PLACEHOLDER>\[(\d+)\]", text)
        for index in image_indices:
            text = text.replace(f"{IMAGE_PLACEHOLDER}[{index}]", '')
        text = text.replace(IMAGE_PLACEHOLDER, '')
        return text
    
    def _get_images(self):
        if self.caption_aux_images:
            image_table = self.aux_image_table
        else:
            image_table = self.image_table
        
        images = []
        filter_dict = {'scoring_label': True, 'final_score': {'$ne': None}}
        if not self.caption_aux_images:
            filter_dict['split_id'] = {'$ne': None}
        
        if self.caption_model_name is None:
            filter_dict['caption'] = {'$eq': None}
            for image in image_table.find(filter_dict, limit=self.batch_size):
                images.append(image)
        else:
            for image in image_table.find(filter_dict):
                if 'all_captions' not in image or image['all_captions'] is None or self.caption_model_name not in image['all_captions'] or image['all_captions'][self.caption_model_name] is None:
                    images.append(image)
                    if self.batch_size > 0 and len(images) == self.batch_size:
                        break
        return images
    
    def _get_images_with_given_query_ids(self, query_ids):
        if self.caption_aux_images:
            image_table = self.aux_image_table
        else:
            image_table = self.image_table
        
        images = []
        filter_dict = {'scoring_label': True, 'final_score': {'$ne': None}}
        if not self.caption_aux_images:
            filter_dict['split_id'] = {'$ne': None}
        
        if self.caption_model_name is None:
            filter_dict['caption'] = {'$eq': None}
            for query_id in query_ids:
                filter_dict['query_id'] = query_id
                for image in image_table.find(filter_dict, limit=self.batch_size):
                    images.append(image)
                if self.batch_size > 0 and len(images) >= self.batch_size:
                    break
        else:
            for query_id in query_ids:
                filter_dict['query_id'] = query_id
                for image in image_table.find(filter_dict):
                    if 'all_captions' not in image or image['all_captions'] is None or self.caption_model_name not in image['all_captions'] or image['all_captions'][self.caption_model_name] is None:
                        images.append(image)
                if self.batch_size > 0 and len(images) >= self.batch_size:
                    break
        
        if self.batch_size > 0:
            images = images[:self.batch_size]
        return images
    
    def _watch_single_cycle(self, *args, **kwargs) -> Tuple[int]:
        query_ids = kwargs.pop('query_ids', None)
        if query_ids is not None:
            images = self._get_images_with_given_query_ids(query_ids=query_ids)
        else:
            images = self._get_images()
        
        if self.caption_aux_images:
            return self._watch_single_cycle_aux(images=images)
        else:
            return self._watch_single_cycle_standard(images=images)
        
    def _watch_single_cycle_standard(self, images: List[dict]) -> Tuple[int, int]:        
        for image in images:
            query_log_dir = os.path.join(self.log_dir, f"query-{image['query_id']}")
            if not os.path.exists(query_log_dir):
                os.makedirs(query_log_dir, exist_ok=True)
                
            cleaned_webpage_splits = self.webpage_table.find_one({'query_id': image['query_id'], 'webpage_id': image['webpage_id']})['cleaned_webpage_splits']
            full_webpage = ''
            for split in cleaned_webpage_splits:
                full_webpage += split['text']
            
            target_placeholder = f"{IMAGE_PLACEHOLDER}[{image['image_id']}]"
            full_webpage_splits = full_webpage.split(target_placeholder)
            assert len(full_webpage_splits) == 2, f"{full_webpage_splits} {image}"
            
            context_above = self._remove_image_placeholders(text=full_webpage_splits[0])
            context_below = self._remove_image_placeholders(text=full_webpage_splits[1])
            
            if self.caption_model_name is None:
                log_file = os.path.join(query_log_dir, f"webpage-{image['webpage_id']}.jsonl")
            else:
                log_file = os.path.join(query_log_dir, f"webpage-{image['webpage_id']}-{self.caption_model_name}.jsonl")
            
            task_id = f"{image['query_id']}-{image['webpage_id']}-{image['image_id']}"
            if self.caption_model_name is not None:
                task_id += f"-{self.caption_model_name}"
            self.manager.add_task(
                self.caption_agent.get_image_caption_single,
                log_file,
                task_id,
                image_url=image['cached_image_url'] if not self.use_orig_image else image['image_url'],
                ref_text=None,
                context=context_above + IMAGE_PLACEHOLDER + context_below,
                query_id=image['query_id'],
                webpage_id=image['webpage_id'],
                image_id=image['image_id']
            )
            
        results = self.manager.execute_tasks()
        self.manager.clear_tasks()
        
        for result in results:
            if result['result'] is not None and result['success']:
                filter_dict = {
                    'query_id': result['kwargs']['query_id'],
                    'webpage_id': result['kwargs']['webpage_id'],
                    'image_id': result['kwargs']['image_id']
                }
                
                if self.caption_model_name is None:
                    self.image_table.update_one(
                        filter=filter_dict,
                        update={'$set': {'caption_label': True, 'caption': result['result']['response']}}
                    )
                else:
                    image = self.image_table.find_one(filter=filter_dict)
                    if 'all_captions' not in image or image['all_captions'] is None:
                        all_captions = {}
                    else:
                        all_captions = image['all_captions']
                    all_captions[self.caption_model_name] = result['result']['response']
                    self.image_table.update_one(
                        filter=filter_dict,
                        update={'$set': {'caption_label': True, 'all_captions': all_captions}}
                    )
        
        return sum([int(result['result'] is not None and result['success']) for result in results]), len(results)
    
    def _watch_single_cycle_aux(self, images: List[dict]) -> Tuple[int, int]:
        for image in images:
            query_log_dir = os.path.join(self.log_dir, f"query-{image['query_id']}")
            if not os.path.exists(query_log_dir):
                os.makedirs(query_log_dir, exist_ok=True)
                
            query_content = self.query_table.find_one({'query_id': image['query_id']}, {'query_content': 1})['query_content']
            
            if self.caption_model_name is None:
                log_file = os.path.join(query_log_dir, f"aux_images.jsonl")
            else:
                log_file = os.path.join(query_log_dir, f"aux_images-{self.caption_model_name}.jsonl")
            
            task_id = f"{image['query_id']}-{image['image_id']}"
            if self.caption_model_name is not None:
                task_id += f"-{self.caption_model_name}"
            self.manager.add_task(
                self.caption_agent.get_image_caption_single,
                log_file,
                task_id,
                image_url=image['cached_image_url'] if not self.use_orig_image else image['image_url'],
                ref_text=query_content,
                context=None,
                query_id=image['query_id'],
                image_id=image['image_id']
            )
            
        results = self.manager.execute_tasks()
        self.manager.clear_tasks()
        
        for result in results:
            if result['result'] is not None and result['success']:
                filter_dict = {
                    'query_id': result['kwargs']['query_id'],
                    'image_id': result['kwargs']['image_id']
                }
                if self.caption_model_name is None:
                    self.aux_image_table.update_one(
                        filter=filter_dict,
                        update={'$set': {'caption_label': True, 'caption': result['result']['response']}}
                    )
                else:
                    aux_image = self.aux_image_table.find_one(filter=filter_dict)
                    if 'all_captions' not in aux_image or aux_image['all_captions'] is None:
                        all_captions = {}
                    else:
                        all_captions = aux_image['all_captions']
                    all_captions[self.caption_model_name] = result['result']['response']
                    self.aux_image_table.update_one(
                        filter=filter_dict,
                        update={'$set': {'caption_label': True, 'all_captions': all_captions}}
                    )
        
        return sum([int(result['result'] is not None and result['success']) for result in results]), len(results)
