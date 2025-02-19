import os
import requests
from typing import Any
from src.task_watcher import TASK_WATCHER
from src.task_watcher.watcher import TaskWatcher
from src.image_handler.image_handler import convert_image_url_to_pil_image
from src.image_handler.image_cache import build_image_cache
from src.utils import MultithreadManager


@TASK_WATCHER.register_module
class CacheImageWatcher(TaskWatcher):
    def __init__(
        self,
        mongodb_url: str,
        database_name: str,
        image_table_name: str = 'images',
        batch_size: int = 0,
        image_relative_dir: str = None,
        send_image_url: bool = False,
        image_cache_config: dict = {},
        multi_thread_config: dict = {},
        max_retry: int = -1
    ) -> None:
        super().__init__(mongodb_url=mongodb_url, database_name=database_name, batch_size=batch_size)
        self.image_table = self.database[image_table_name]
        self.image_cache = build_image_cache(image_cache_config)
        self.manager = MultithreadManager(**multi_thread_config)
        self.image_relative_dir = image_relative_dir
        self.send_image_url = send_image_url
        self.max_retry = max_retry
        
    def _convert_image_and_save(self, image_url: str, relative_dir: str, image_id: Any, **kwargs):
        # send a relatively smaller http packet for remote image cache
        if self.send_image_url:
            relative_basename = os.path.join(relative_dir, f"{image_id}")
            cached_image_url = self.image_cache.save(pil_image=None, relative_path=None, image_url=image_url, relative_basename=relative_basename)
            if cached_image_url is not None:
                relative_path = cached_image_url[cached_image_url.find(relative_basename):]
                return {
                    "relative_path": relative_path,
                    "cached_image_url": cached_image_url 
                }
            else:
                return None
        else:
            pil_image = convert_image_url_to_pil_image(image_url=image_url)
            if pil_image is not None:
                relative_path = os.path.join(relative_dir, f"{image_id}.{pil_image.format.lower()}")
                cached_image_url = self.image_cache.save(pil_image=pil_image, relative_path=relative_path)
                return {
                    "relative_path": relative_path,
                    "cached_image_url": cached_image_url
                }
            else:
                return None
        
    def _watch_single_cycle(self, *args, **kwargs):
        # for safety
        self.image_table.update_many({'retry': None}, {'$set': {'retry': 0}})

        for image in self.image_table.find({'valid': None, 'cache_label': False}, limit=self.batch_size):
            if 'webpage_id' in image:
                task_id = f"{image['query_id']}-{image['webpage_id']}-{image['image_id']}"
                relative_dir = os.path.join(self.image_relative_dir, f"query-{image['query_id']}", f"webpage-{image['webpage_id']}")
                webpage_id = image['webpage_id']
            else:
                task_id = f"{image['query_id']}-{image['image_id']}"
                relative_dir = os.path.join(self.image_relative_dir, f"query-{image['query_id']}")
                webpage_id = None
                
            self.manager.add_task(
                self._convert_image_and_save,
                None,
                task_id,
                image_url=image['image_url'],
                relative_dir=relative_dir,
                query_id=image['query_id'],
                webpage_id=webpage_id,
                image_id=image['image_id']
            )
            
        results = self.manager.execute_tasks()
        self.manager.clear_tasks()
        
        for result in results:
            filter_dict = {'query_id': result['kwargs']['query_id'], 'image_id': result['kwargs']['image_id']}
            if result['kwargs']['webpage_id'] is not None:
                filter_dict['webpage_id'] = result['kwargs']['webpage_id']
                
            if result['result'] is not None and result['success']:
                update = {'$set': {'cache_label': True, 'cached_image_url': result['result']['cached_image_url'], 'relative_path': result['result']['relative_path'], 'valid': True}}
                self.image_table.update_one(
                    filter=filter_dict,
                    update=update
                )
            else:
                update = {'$inc': {'retry': 1}}
                self.image_table.update_one(
                    filter=filter_dict,
                    update=update
                )
        
        self.image_table.update_many({'retry': {'$gte': self.max_retry}}, {'$set': {'valid': False}})
        
        return sum([int(result['result'] is not None and result['success']) for result in results]), len(results)
    