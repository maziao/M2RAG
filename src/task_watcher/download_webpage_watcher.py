import os
from typing import Optional
from src.task_watcher import TASK_WATCHER
from src.task_watcher.watcher import TaskWatcher
from src.crawler import build_crawler


@TASK_WATCHER.register_module
class DownloadWebpageWatcher(TaskWatcher):
    def __init__(
        self,
        mongodb_url: str,
        database_name: str,
        batch_size: int = 0,
        crawler_config: dict = {},
        max_retry: int = 3,
        log_dir: Optional[str] = None,
    ) -> None:
        super().__init__(mongodb_url=mongodb_url, database_name=database_name, batch_size=batch_size)
        self.webpage_table = self.database['webpages']
        self.crawler = build_crawler(crawler_config)
        self.max_retry = max_retry
        
        if log_dir is not None:
            self.log_dir = os.path.join(log_dir, "download_webpage")
            if not os.path.exists(self.log_dir):
                os.makedirs(self.log_dir, exist_ok=True)
        else:
            self.log_dir = None
        
    def _watch_single_cycle(self, *args, **kwargs):
        # for safety
        self.webpage_table.update_many({'retry': None}, {'$set': {'retry': 0}})
        
        for webpage in self.webpage_table.find({'webpage_label': False, 'retry': {'$lt': self.max_retry}}, limit=self.batch_size):
            self.crawler.manager.add_task(
                self.crawler.download_webpage,
                os.path.join(self.log_dir, f"{webpage['query_id']}", f"{webpage['webpage_id']}.json") if self.log_dir is not None else None,
                f"{webpage['query_id']}-{webpage['webpage_id']}",
                url=webpage['webpage_url'],
                query_id=webpage['query_id'],
                webpage_id=webpage['webpage_id']
            )
        
        results = self.crawler.manager.execute_tasks()
        self.crawler.manager.clear_tasks()
        
        for result in results:
            if result['result'] is not None and result['success']:
                filter_dict = {'query_id': result['kwargs']['query_id'], 'webpage_id': result['kwargs']['webpage_id']}                
                update = {'$set': {'webpage_label': True, 'webpage_content': result['result']['content']}}
                if 'tokens' in result['result']['usage']:
                    update['$set']['webpage_tokens'] = result['result']['usage']['tokens']
                self.webpage_table.update_many(
                    filter=filter_dict,
                    update=update
                )
            else:
                self.webpage_table.update_many(
                    filter={'query_id': result['kwargs']['query_id'], 'webpage_id': result['kwargs']['webpage_id']},
                    update={'$inc': {'retry': 1}}
                )
        return sum([int(result['success'] and result['result'] is not None) for result in results]), len(results)
    