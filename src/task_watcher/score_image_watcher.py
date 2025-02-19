from src.task_watcher import TASK_WATCHER
from src.task_watcher.watcher import TaskWatcher
from src.image_handler.image_scorer import build_image_scorer
from src.utils import MultithreadManager


@TASK_WATCHER.register_module
class ScoreImageWatcher(TaskWatcher):
    def __init__(
        self,
        mongodb_url: str,
        database_name: str,
        batch_size: int = 0,
        image_scorer_config: dict = {},
        score_aux_images: bool = False,
        multi_thread_config: dict = {}
    ) -> None:
        super().__init__(mongodb_url=mongodb_url, database_name=database_name, batch_size=batch_size)
        self.query_table = self.database['queries']
        self.webpage_table = self.database['webpages']
        self.image_table = self.database['images']
        self.aux_image_table = self.database['aux_images']
        self.image_scorer = build_image_scorer(image_scorer_config)
        self.score_aux_images = score_aux_images
        self.manager = MultithreadManager(**multi_thread_config)
    
    def _watch_single_cycle(self, *args, **kwargs):
        if self.score_aux_images:
            return self._watch_single_cycle_aux(*args, **kwargs)
        else:
            return self._watch_single_cycle_standard(*args, **kwargs)
        
    def _watch_single_cycle_standard(self, *args, **kwargs):
        for webpage in self.webpage_table.find({'clean_label': True, 'image_dedup_label': True, 'image_score_label': False}, limit=self.batch_size):
            query = self.query_table.find_one({'query_id': webpage['query_id']})['query_content']
            image_urls = []
            image_ids = []
            for image in self.image_table.find({'query_id': webpage['query_id'], 'webpage_id': webpage['webpage_id'], 'cache_label': True, 'duplicate_label': False}):
                image_urls.append(image['cached_image_url'])
                image_ids.append(image['image_id'])
            self.manager.add_task(
                self.image_scorer.get_image_scores,
                None,
                f"{webpage['query_id']}-{webpage['webpage_id']}",
                image_urls=image_urls,
                reference_text=query,
                query_id=webpage['query_id'],
                webpage_id=webpage['webpage_id'],
                image_ids=image_ids
            )

        results = self.manager.execute_tasks()
        self.manager.clear_tasks()
        
        for result in results:
            if result['result'] is not None and result['success']:
                num_valid_images = 0
                for image_id, image_score in zip(result['kwargs']['image_ids'], result['result']):
                    final_score = image_score.pop('score')
                    if final_score is not None:
                        num_valid_images += 1
                    self.image_table.update_one(
                        filter={'query_id': result['kwargs']['query_id'], 'webpage_id': result['kwargs']['webpage_id'], 'image_id': image_id},
                        update={'$set': {'scoring_label': True, 'scores': image_score, 'final_score': final_score}}
                    )
                self.webpage_table.update_one(
                    filter={'query_id': result['kwargs']['query_id'], 'webpage_id': result['kwargs']['webpage_id']},
                    update={'$set': {'image_score_label': True, 'num_valid_images': num_valid_images}}
                )
        return sum([int(result['result'] is not None and result['success']) for result in results]), len(results)
    
    def _watch_single_cycle_aux(self, *args, **kwargs):
        for aux_image in self.aux_image_table.find({'cache_label': True, 'scoring_label': False}, limit=self.batch_size):
            query = self.query_table.find_one({'query_id': aux_image['query_id']})['query_content']
            self.manager.add_task(
                self.image_scorer.get_image_scores,
                None,
                f"{aux_image['query_id']}-{aux_image['image_id']}",
                image_urls=[aux_image['cached_image_url']],
                reference_text=query,
                query_id=aux_image['query_id'],
                image_id=aux_image['image_id']
            )

        results = self.manager.execute_tasks()
        self.manager.clear_tasks()
        
        for result in results:
            if result['result'] is not None and result['success']:
                final_score = result['result'][0].pop('score')
                self.aux_image_table.update_one(
                    filter={'query_id': result['kwargs']['query_id'], 'image_id': result['kwargs']['image_id']},
                    update={'$set': {'scoring_label': True, 'scores': result['result'][0], 'final_score': final_score}}
                )
        return sum([int(result['result'] is not None and result['success']) for result in results]), len(results)
    