from src.utils import MultithreadManager
from src.task_watcher import TASK_WATCHER
from src.task_watcher.watcher import TaskWatcher
from src.webpage_cleaner import build_webpage_cleaner


@TASK_WATCHER.register_module
class CleanWebpageWatcher(TaskWatcher):
    def __init__(
        self,
        mongodb_url: str,
        database_name: str,
        batch_size: int = 0,
        webpage_cleaner_config: dict = {},
        multi_thread_config: dict = {}
    ) -> None:
        super().__init__(mongodb_url=mongodb_url, database_name=database_name, batch_size=batch_size)
        self.query_table = self.database['queries']
        self.webpage_table = self.database['webpages']
        self.image_table = self.database['images']
        self.webpage_cleaner = build_webpage_cleaner(webpage_cleaner_config)
        self.manager = MultithreadManager(**multi_thread_config)
        
    def _watch_single_cycle(self, *args, **kwargs):
        for webpage in self.webpage_table.find({'webpage_label': True, 'clean_label': False}, limit=self.batch_size):
            query_content = self.query_table.find_one({'query_id': webpage['query_id']})['query_content']
            
            self.manager.add_task(
                self.webpage_cleaner.clean,
                None,
                f"{webpage['query_id']}-{webpage['webpage_id']}",
                text=webpage['webpage_content'],
                ref_text=query_content,
                query_id=webpage['query_id'],
                webpage_id=webpage['webpage_id']
            )
        
        results = self.manager.execute_tasks()
        self.manager.clear_tasks()
        
        for result in results:
            if result['result'] is not None and result['success']:
                update = {
                    '$set': {
                        'clean_label': True,
                        'char_len_trend': result['result']['char_len_trend'],
                        'cleaned_webpage_splits': result['result']['cleaned_webpage_splits'],
                        'images': result['result']['images']
                    }
                }
                self.webpage_table.update_one(
                    filter={'query_id': result['kwargs']['query_id'], 'webpage_id': result['kwargs']['webpage_id']},
                    update=update
                )
                for image in result['result']['images']:
                    self.image_table.insert_one({
                        'query_id': result['kwargs']['query_id'],
                        'webpage_id': result['kwargs']['webpage_id'],
                        'split_id': image['split_id'],
                        'image_id': image['image_id'],
                        'image_url': image['image_url'],
                        'valid': None,
                        'retry': 0,
                        'cache_label': False,
                        'cached_image_url': None,
                        'relative_path': None,
                        'duplicate_label': None,
                        'scoring_label': False,
                        'scores': None,
                        'final_score': None,
                        'caption_label': False,
                        'caption': None,
                        'all_captions': {}
                    })
        return sum([int(result['result'] is not None and result['success']) for result in results]), len(results)
