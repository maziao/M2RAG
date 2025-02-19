import os
import re
from typing import Optional, Tuple
from src.task_watcher import TASK_WATCHER
from src.task_watcher.watcher import TaskWatcher
from src.image_handler.image_dedup import build_image_dedup


@TASK_WATCHER.register_module
class ImageDeduplicateWatcher(TaskWatcher):
    def __init__(
        self,
        mongodb_url: Optional[str] = None,
        database_name: Optional[str] = None,
        batch_size: int = 0,
        interval_on_empty_cycle: int = 60,
        image_root: Optional[str] = None,
        image_dedup_config: dict = {}
    ) -> None:
        super().__init__(mongodb_url, database_name, batch_size, interval_on_empty_cycle)
        self.webpage_table = self.database['webpages']
        self.image_table = self.database['images']
        self.dedup_agent = build_image_dedup(image_dedup_config)
        if image_root is None:
            image_root = os.environ.get('IMAGE_ROOT')
        assert image_root is not None
        self.image_root = image_root
        
    def _watch_single_cycle(self, *args, **kwargs) -> Tuple[int, int]:
        counter = 0
        for webpage in self.webpage_table.find({'clean_label': True, 'image_dedup_label': False}, limit=self.batch_size):
            image = self.image_table.find_one({'query_id': webpage['query_id'], 'webpage_id': webpage['webpage_id'], 'cache_label': True})
            
            # no valid images
            if image is None:
                self.webpage_table.update_one(
                    filter={'query_id': webpage['query_id'], 'webpage_id': webpage['webpage_id']},
                    update={'$set': {'image_dedup_label': True}}
                )
                continue
            
            local_image_relative_path = image['relative_path']
            image_dir = os.path.join(self.image_root, os.path.dirname(local_image_relative_path))
            
            result = self.dedup_agent.get_duplicate_images(
                image_dir=image_dir
            )
            
            for image in result:
                image_id = int(re.findall(r"\d+", image)[0])
                self.image_table.update_one(
                    filter={'query_id': webpage['query_id'], 'webpage_id': webpage['webpage_id'], 'image_id': image_id},
                    update={'$set': {'duplicate_label': True}}
                )
            self.image_table.update_many(
                filter={'query_id': webpage['query_id'], 'webpage_id': webpage['webpage_id'], 'duplicate_label': None},
                update={'$set': {'duplicate_label': False}}
            )
            self.webpage_table.update_one(
                filter={'query_id': webpage['query_id'], 'webpage_id': webpage['webpage_id']},
                update={'$set': {'image_dedup_label': True}}
            )
            counter += 1
        return counter, counter
